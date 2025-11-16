// Function: sub_2851110
// Address: 0x2851110
//
void __fastcall sub_2851110(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // r13
  __int64 v10; // rcx
  unsigned __int64 v11; // r12
  _QWORD *v12; // rcx
  _QWORD *v13; // r15
  unsigned __int64 *v14; // r14
  unsigned __int64 v15; // rdi
  int v16; // r15d
  __int64 v17; // [rsp+8h] [rbp-48h]
  unsigned __int64 v18[7]; // [rsp+18h] [rbp-38h] BYREF

  v17 = a1 + 16;
  v7 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 8u, v18, a6);
  v8 = *(_QWORD **)a1;
  v9 = v7;
  v10 = *(unsigned int *)(a1 + 8);
  v11 = *(_QWORD *)a1 + v10 * 8;
  if ( *(_QWORD *)a1 != v11 )
  {
    v12 = &v7[v10];
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
    while ( v7 != v12 );
    v13 = *(_QWORD **)a1;
    v11 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v11 )
    {
      do
      {
        v14 = *(unsigned __int64 **)(v11 - 8);
        v11 -= 8LL;
        if ( v14 )
        {
          v15 = v14[8];
          if ( (unsigned __int64 *)v15 != v14 + 10 )
            _libc_free(v15);
          if ( (unsigned __int64 *)*v14 != v14 + 2 )
            _libc_free(*v14);
          j_j___libc_free_0((unsigned __int64)v14);
        }
      }
      while ( v13 != (_QWORD *)v11 );
      v11 = *(_QWORD *)a1;
    }
  }
  v16 = v18[0];
  if ( v17 != v11 )
    _libc_free(v11);
  *(_QWORD *)a1 = v9;
  *(_DWORD *)(a1 + 12) = v16;
}
