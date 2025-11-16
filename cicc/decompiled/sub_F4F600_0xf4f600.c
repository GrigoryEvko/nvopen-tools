// Function: sub_F4F600
// Address: 0xf4f600
//
__int64 __fastcall sub_F4F600(__int64 a1, __int64 a2, __int64 a3)
{
  char v4; // r10
  __int64 v5; // r9
  int v6; // r11d
  unsigned int v7; // r8d
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  int v14; // eax
  int v15; // ebx
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rax

  v4 = *(_BYTE *)(a3 + 8) & 1;
  if ( v4 )
  {
    v5 = a3 + 16;
    v6 = 15;
  }
  else
  {
    v12 = *(unsigned int *)(a3 + 24);
    v5 = *(_QWORD *)(a3 + 16);
    if ( !(_DWORD)v12 )
      goto LABEL_12;
    v6 = v12 - 1;
  }
  v7 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( a2 == *v8 )
  {
LABEL_4:
    v10 = 256;
    if ( !v4 )
      v10 = 16LL * *(unsigned int *)(a3 + 24);
    if ( v8 != (__int64 *)(v5 + v10) )
      return v8[1];
    return a1;
  }
  v14 = 1;
  if ( v9 == -4096 )
  {
    if ( v4 )
    {
      v13 = 256;
      goto LABEL_13;
    }
    v12 = *(unsigned int *)(a3 + 24);
LABEL_12:
    v13 = 16 * v12;
LABEL_13:
    v8 = (__int64 *)(v5 + v13);
    goto LABEL_4;
  }
  while ( 1 )
  {
    v15 = v14 + 1;
    v7 = v6 & (v14 + v7);
    v16 = (__int64 *)(v5 + 16LL * v7);
    v17 = *v16;
    if ( a2 == *v16 )
      break;
    v14 = v15;
    if ( v17 == -4096 )
    {
      if ( v4 )
        v19 = 256;
      else
        v19 = 16LL * *(unsigned int *)(a3 + 24);
      v16 = (__int64 *)(v5 + v19);
      break;
    }
  }
  v18 = 256;
  if ( !v4 )
    v18 = 16LL * *(unsigned int *)(a3 + 24);
  if ( v16 != (__int64 *)(v5 + v18) )
    return v16[1];
  return a1;
}
