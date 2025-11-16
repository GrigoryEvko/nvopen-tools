// Function: sub_1A4EF50
// Address: 0x1a4ef50
//
__int64 __fastcall sub_1A4EF50(__int64 *a1, __int64 a2)
{
  __int64 v2; // r9
  int v4; // edi
  __int64 v5; // rcx
  int v6; // edx
  int v7; // r11d
  __int64 v8; // rbx
  __int64 *v9; // rax
  __int64 v10; // r10
  __int64 v11; // rdx
  unsigned int v12; // r9d
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // r9d

  v2 = *a1;
  v4 = *(_BYTE *)(*a1 + 8) & 1;
  if ( v4 )
  {
    v5 = v2 + 16;
    v6 = 15;
  }
  else
  {
    v14 = *(unsigned int *)(v2 + 24);
    v5 = *(_QWORD *)(v2 + 16);
    if ( !(_DWORD)v14 )
    {
      v16 = v5 + 16 * v14;
      v17 = 16LL * *(unsigned int *)(v2 + 24);
      v18 = 0;
      if ( v16 != v5 + v17 )
        LOBYTE(v18) = *(_QWORD *)a1[1] != *(_QWORD *)(v16 + 8);
      return v18;
    }
    v6 = v14 - 1;
  }
  v7 = 1;
  LODWORD(v8) = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v5 + 16LL * (unsigned int)v8);
  v10 = *v9;
  if ( a2 != *v9 )
  {
    while ( v10 != -8 )
    {
      v8 = v6 & (unsigned int)(v7 + v8);
      v9 = (__int64 *)(v5 + 16 * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_4;
      ++v7;
    }
    if ( (_BYTE)v4 )
      v15 = 256;
    else
      v15 = 16LL * *(unsigned int *)(v2 + 24);
    v9 = (__int64 *)(v5 + v15);
  }
LABEL_4:
  v11 = 256;
  if ( !(_BYTE)v4 )
    v11 = 16LL * *(unsigned int *)(v2 + 24);
  v12 = 0;
  if ( v9 != (__int64 *)(v5 + v11) )
    LOBYTE(v12) = *(_QWORD *)a1[1] != v9[1];
  return v12;
}
