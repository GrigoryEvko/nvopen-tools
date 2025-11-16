// Function: sub_31FC930
// Address: 0x31fc930
//
void __fastcall sub_31FC930(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  _QWORD *v7; // r12
  __int64 v8; // rcx
  unsigned __int64 v9; // r15
  _QWORD *v10; // rdx
  _QWORD *v11; // rcx
  __int64 v12; // r13
  __int64 v13; // rax
  unsigned __int64 *v14; // rax
  unsigned __int64 v15; // r14
  int v16; // r14d
  unsigned __int64 v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 0x10u, v17, a6);
  v7 = v6;
  v8 = 2LL * *(unsigned int *)(a1 + 8);
  v9 = *(_QWORD *)a1 + v8 * 8;
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = (_QWORD *)(*(_QWORD *)a1 + 8LL);
    v11 = &v6[v8];
    do
    {
      if ( v6 )
      {
        *v6 = *(v10 - 1);
        v6[1] = *v10;
        *v10 = 0;
      }
      v6 += 2;
      v10 += 2;
    }
    while ( v11 != v6 );
    v12 = *(_QWORD *)a1;
    v9 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v13 = *(_QWORD *)(v9 - 8);
        v9 -= 16LL;
        if ( v13 )
        {
          if ( (v13 & 4) != 0 )
          {
            v14 = (unsigned __int64 *)(v13 & 0xFFFFFFFFFFFFFFF8LL);
            v15 = (unsigned __int64)v14;
            if ( v14 )
            {
              if ( (unsigned __int64 *)*v14 != v14 + 2 )
                _libc_free(*v14);
              j_j___libc_free_0(v15);
            }
          }
        }
      }
      while ( v12 != v9 );
      v9 = *(_QWORD *)a1;
    }
  }
  v16 = v17[0];
  if ( a1 + 16 != v9 )
    _libc_free(v9);
  *(_QWORD *)a1 = v7;
  *(_DWORD *)(a1 + 12) = v16;
}
