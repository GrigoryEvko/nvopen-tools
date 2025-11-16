// Function: sub_2FC1040
// Address: 0x2fc1040
//
_QWORD *__fastcall sub_2FC1040(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // rcx
  unsigned __int64 v10; // r12
  _QWORD *v11; // rcx
  unsigned __int64 *v12; // r15
  unsigned __int64 v13; // r13
  unsigned __int64 v14; // r14
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  int v17; // r13d
  _QWORD *v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h]
  _QWORD *v21; // [rsp+18h] [rbp-48h]
  unsigned __int64 v22[7]; // [rsp+28h] [rbp-38h] BYREF

  v20 = a1 + 16;
  v7 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 8u, v22, a6);
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
    while ( v7 != v11 );
    v10 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    v21 = *(_QWORD **)a1;
    if ( *(_QWORD *)a1 != v10 )
    {
      do
      {
        v12 = *(unsigned __int64 **)(v10 - 8);
        v10 -= 8LL;
        if ( v12 )
        {
          sub_2E0AFD0((__int64)v12);
          v13 = v12[12];
          if ( v13 )
          {
            v14 = *(_QWORD *)(v13 + 16);
            while ( v14 )
            {
              sub_2FBF390(*(_QWORD *)(v14 + 24));
              v15 = v14;
              v14 = *(_QWORD *)(v14 + 16);
              j_j___libc_free_0(v15);
            }
            j_j___libc_free_0(v13);
          }
          v16 = v12[8];
          if ( (unsigned __int64 *)v16 != v12 + 10 )
            _libc_free(v16);
          if ( (unsigned __int64 *)*v12 != v12 + 2 )
            _libc_free(*v12);
          j_j___libc_free_0((unsigned __int64)v12);
        }
      }
      while ( v21 != (_QWORD *)v10 );
      v10 = *(_QWORD *)a1;
    }
  }
  v17 = v22[0];
  if ( v20 != v10 )
    _libc_free(v10);
  *(_DWORD *)(a1 + 12) = v17;
  *(_QWORD *)a1 = v19;
  return v19;
}
