// Function: sub_2F7AA70
// Address: 0x2f7aa70
//
__int64 __fastcall sub_2F7AA70(__int64 a1)
{
  unsigned int v2; // r15d
  int v3; // r14d
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  unsigned __int64 v6; // rdi
  int v7; // eax
  unsigned int v9; // edx
  __int64 v10; // rbx
  _QWORD *v11; // rdi
  unsigned int v12; // ebx
  unsigned __int64 v13; // rdi
  _QWORD *v14; // rax
  __int64 v15; // rdx
  _QWORD *i; // rdx
  void *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  _QWORD *v21; // rax

  if ( (_BYTE)qword_5024F08 )
  {
    v17 = sub_CB72A0();
    sub_2F7A520(a1, (__int64)v17, 0, v18, v19, v20);
  }
  v2 = *(_DWORD *)(a1 + 24);
  v3 = *(_DWORD *)(a1 + 16);
  if ( !v2 )
  {
    if ( !v3 )
      goto LABEL_33;
    v9 = v3 - 1;
    if ( v3 == 1 )
    {
      v11 = *(_QWORD **)(a1 + 8);
      LODWORD(v10) = 64;
LABEL_20:
      sub_C7D6A0((__int64)v11, 32LL * v2, 8);
      v12 = 4 * (int)v10 / 3u;
      v13 = ((((((((((v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 2)
                 | (v12 + 1)
                 | ((unsigned __int64)(v12 + 1) >> 1)) >> 4)
               | (((v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 2)
               | (v12 + 1)
               | ((unsigned __int64)(v12 + 1) >> 1)) >> 8)
             | (((((v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 2) | (v12 + 1)
                                                                         | ((unsigned __int64)(v12 + 1) >> 1)) >> 4)
             | (((v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 2)
             | (v12 + 1)
             | ((unsigned __int64)(v12 + 1) >> 1)) >> 16)
           | (((((((v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 2) | (v12 + 1)
                                                                         | ((unsigned __int64)(v12 + 1) >> 1)) >> 4)
             | (((v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 2)
             | (v12 + 1)
             | ((unsigned __int64)(v12 + 1) >> 1)) >> 8)
           | (((((v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 2) | (v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 4)
           | (((v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 2)
           | (v12 + 1)
           | ((unsigned __int64)(v12 + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 24) = v13;
      v14 = (_QWORD *)sub_C7D670(32 * v13, 8);
      v15 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = v14;
      for ( i = &v14[4 * v15]; i != v14; v14 += 4 )
      {
        if ( v14 )
          *v14 = -4096;
      }
      return 0;
    }
    v7 = 0;
    goto LABEL_17;
  }
  v4 = *(_QWORD **)(a1 + 8);
  v5 = &v4[4 * v2];
  do
  {
    if ( *v4 != -4096 && *v4 != -8192 )
    {
      v6 = v4[1];
      if ( v6 )
        j_j___libc_free_0(v6);
    }
    v4 += 4;
  }
  while ( v5 != v4 );
  v7 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    if ( v7 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 8), 32LL * v2, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return 0;
    }
LABEL_33:
    *(_QWORD *)(a1 + 16) = 0;
    return 0;
  }
  v10 = 64;
  v9 = v3 - 1;
  if ( v3 != 1 )
  {
LABEL_17:
    _BitScanReverse(&v9, v9);
    v10 = (unsigned int)(1 << (33 - (v9 ^ 0x1F)));
    if ( (int)v10 < 64 )
      v10 = 64;
  }
  v11 = *(_QWORD **)(a1 + 8);
  if ( (_DWORD)v10 != v7 )
    goto LABEL_20;
  *(_QWORD *)(a1 + 16) = 0;
  v21 = &v11[4 * v10];
  do
  {
    if ( v11 )
      *v11 = -4096;
    v11 += 4;
  }
  while ( v21 != v11 );
  return 0;
}
