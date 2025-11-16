// Function: sub_2B486F0
// Address: 0x2b486f0
//
_QWORD *__fastcall sub_2B486F0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // rcx
  unsigned __int64 v10; // r12
  _QWORD *v11; // rcx
  _QWORD *v12; // r13
  __int64 v13; // r15
  __int64 v14; // r14
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  int v17; // r13d
  _QWORD *v19; // [rsp+0h] [rbp-50h]
  __int64 v20; // [rsp+8h] [rbp-48h]
  unsigned __int64 v21[7]; // [rsp+18h] [rbp-38h] BYREF

  v20 = a1 + 16;
  v7 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 8u, v21, a6);
  v8 = *(_QWORD **)a1;
  v19 = v7;
  v9 = *(unsigned int *)(a1 + 8);
  v10 = *(_QWORD *)a1 + v9 * 8;
  if ( v8 != &v8[v9] )
  {
    v11 = &v7[v9];
    do
    {
      if ( v7 )
      {
        *v7 = *v8;
        *v8 = 0;
      }
      ++v7;
      ++v8;
    }
    while ( v11 != v7 );
    v12 = *(_QWORD **)a1;
    v10 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v10 )
    {
      do
      {
        v13 = *(_QWORD *)(v10 - 8);
        v10 -= 8LL;
        if ( v13 )
        {
          v14 = v13 + 160LL * *(_QWORD *)(v13 - 8);
          while ( v13 != v14 )
          {
            v14 -= 160;
            v15 = *(_QWORD *)(v14 + 88);
            if ( v15 != v14 + 104 )
              _libc_free(v15);
            v16 = *(_QWORD *)(v14 + 40);
            if ( v16 != v14 + 56 )
              _libc_free(v16);
          }
          j_j_j___libc_free_0_0(v13 - 8);
        }
      }
      while ( v12 != (_QWORD *)v10 );
      v10 = *(_QWORD *)a1;
    }
  }
  v17 = v21[0];
  if ( v20 != v10 )
    _libc_free(v10);
  *(_DWORD *)(a1 + 12) = v17;
  *(_QWORD *)a1 = v19;
  return v19;
}
