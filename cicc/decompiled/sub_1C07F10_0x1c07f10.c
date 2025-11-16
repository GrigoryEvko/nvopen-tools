// Function: sub_1C07F10
// Address: 0x1c07f10
//
__int64 __fastcall sub_1C07F10(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v7; // r8
  unsigned int v8; // ebx
  __int64 *v9; // rcx
  __int64 v10; // r11
  int v11; // eax
  int v12; // eax
  __int64 v13; // r8
  int v14; // r11d
  unsigned int v15; // edx
  __int64 v16; // rdi
  unsigned int v17; // r8d
  int v19; // ecx
  int v20; // r13d

  v4 = *(unsigned int *)(a1 + 32);
  if ( !(_DWORD)v4 )
    return 0;
  v7 = *(_QWORD *)(a1 + 16);
  v8 = (v4 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
  v9 = (__int64 *)(v7 + 24LL * v8);
  v10 = *v9;
  if ( a2 != *v9 )
  {
    v19 = 1;
    while ( v10 != -8 )
    {
      v20 = v19 + 1;
      v8 = (v4 - 1) & (v19 + v8);
      v9 = (__int64 *)(v7 + 24LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_3;
      v19 = v20;
    }
    return 0;
  }
LABEL_3:
  if ( v9 == (__int64 *)(v7 + 24 * v4) )
    return 0;
  v11 = *(_DWORD *)(a1 + 64);
  if ( v11 )
  {
    v12 = v11 - 1;
    v13 = *(_QWORD *)(a1 + 48);
    v14 = 1;
    v15 = v12 & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
    v16 = *(_QWORD *)(v13 + 8LL * v15);
    if ( a2 == v16 )
    {
LABEL_6:
      v17 = 1;
      *a3 = *((_DWORD *)v9 + 4);
      return v17;
    }
    while ( v16 != -8 )
    {
      v15 = v12 & (v14 + v15);
      v16 = *(_QWORD *)(v13 + 8LL * v15);
      if ( a2 == v16 )
        goto LABEL_6;
      ++v14;
    }
  }
  if ( v9[1] == a4 )
    goto LABEL_6;
  return 0;
}
