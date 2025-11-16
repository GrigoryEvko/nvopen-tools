// Function: sub_210C930
// Address: 0x210c930
//
__int64 __fastcall sub_210C930(__int64 a1)
{
  __int64 v2; // r12
  int v3; // r14d
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  __int64 v6; // rdi
  __int64 v8; // rbx
  unsigned int v9; // r14d
  _QWORD *v10; // rdi
  unsigned int v11; // ebx
  unsigned __int64 v12; // rdi
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  void (__fastcall *v16)(__int64, void *, _QWORD); // rbx
  void *v17; // rax
  _QWORD *v18; // rax

  if ( byte_4FCFB00 )
  {
    v16 = *(void (__fastcall **)(__int64, void *, _QWORD))(*(_QWORD *)a1 + 40LL);
    v17 = sub_16E8CB0();
    v16(a1, v17, 0);
  }
  v2 = *(unsigned int *)(a1 + 184);
  v3 = *(_DWORD *)(a1 + 176);
  if ( !(_DWORD)v2 )
  {
    if ( !v3 )
      goto LABEL_31;
    v9 = v3 - 1;
    if ( !v9 )
    {
      v10 = *(_QWORD **)(a1 + 168);
      LODWORD(v8) = 64;
LABEL_18:
      j___libc_free_0(v10);
      v11 = 4 * (int)v8 / 3u;
      v12 = ((((((((((v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 2)
                 | (v11 + 1)
                 | ((unsigned __int64)(v11 + 1) >> 1)) >> 4)
               | (((v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 2)
               | (v11 + 1)
               | ((unsigned __int64)(v11 + 1) >> 1)) >> 8)
             | (((((v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 2) | (v11 + 1)
                                                                         | ((unsigned __int64)(v11 + 1) >> 1)) >> 4)
             | (((v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 2)
             | (v11 + 1)
             | ((unsigned __int64)(v11 + 1) >> 1)) >> 16)
           | (((((((v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 2) | (v11 + 1)
                                                                         | ((unsigned __int64)(v11 + 1) >> 1)) >> 4)
             | (((v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 2)
             | (v11 + 1)
             | ((unsigned __int64)(v11 + 1) >> 1)) >> 8)
           | (((((v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 2) | (v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 4)
           | (((v11 + 1) | ((unsigned __int64)(v11 + 1) >> 1)) >> 2)
           | (v11 + 1)
           | ((unsigned __int64)(v11 + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 184) = v12;
      v13 = (_QWORD *)sub_22077B0(32 * v12);
      v14 = *(unsigned int *)(a1 + 184);
      *(_QWORD *)(a1 + 176) = 0;
      *(_QWORD *)(a1 + 168) = v13;
      for ( i = &v13[4 * v14]; i != v13; v13 += 4 )
      {
        if ( v13 )
          *v13 = -8;
      }
      return 0;
    }
    goto LABEL_15;
  }
  v4 = *(_QWORD **)(a1 + 168);
  v5 = &v4[4 * v2];
  do
  {
    if ( *v4 != -8 && *v4 != -16 )
    {
      v6 = v4[1];
      if ( v6 )
        j_j___libc_free_0(v6, v4[3] - v6);
    }
    v4 += 4;
  }
  while ( v5 != v4 );
  LODWORD(v2) = *(_DWORD *)(a1 + 184);
  if ( !v3 )
  {
    if ( (_DWORD)v2 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 168));
      *(_QWORD *)(a1 + 168) = 0;
      *(_QWORD *)(a1 + 176) = 0;
      *(_DWORD *)(a1 + 184) = 0;
      return 0;
    }
LABEL_31:
    *(_QWORD *)(a1 + 176) = 0;
    return 0;
  }
  v8 = 64;
  v9 = v3 - 1;
  if ( v9 )
  {
LABEL_15:
    _BitScanReverse(&v9, v9);
    v8 = (unsigned int)(1 << (33 - (v9 ^ 0x1F)));
    if ( (int)v8 < 64 )
      v8 = 64;
  }
  v10 = *(_QWORD **)(a1 + 168);
  if ( (_DWORD)v8 != (_DWORD)v2 )
    goto LABEL_18;
  *(_QWORD *)(a1 + 176) = 0;
  v18 = &v10[4 * v8];
  do
  {
    if ( v10 )
      *v10 = -8;
    v10 += 4;
  }
  while ( v18 != v10 );
  return 0;
}
