// Function: sub_2A3BC20
// Address: 0x2a3bc20
//
__int64 __fastcall sub_2A3BC20(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r15
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 *v12; // rbx
  _QWORD *v13; // r14
  __int64 *v14; // r12
  __int64 v15; // rdx
  __int64 *v16; // rbx
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  int v21; // ebx
  __int64 v23; // [rsp+8h] [rbp-48h]
  unsigned __int64 v24[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = (__int64 *)(a1 + 16);
  v8 = sub_C8D7D0(a1, a1 + 16, a2, 0x90u, v24, a6);
  v12 = *(__int64 **)a1;
  v23 = v8;
  v13 = (_QWORD *)v8;
  v14 = (__int64 *)(*(_QWORD *)a1 + 144LL * *(unsigned int *)(a1 + 8));
  if ( *(__int64 **)a1 != v14 )
  {
    do
    {
      if ( v13 )
      {
        v15 = *v12;
        *v13 = *v12;
        sub_2A3B9E0((__int64)(v13 + 1), (__int64)(v12 + 1), v15, v9, v10, v11);
      }
      v12 += 18;
      v13 += 18;
    }
    while ( v14 != v12 );
    v16 = *(__int64 **)a1;
    v14 = (__int64 *)(*(_QWORD *)a1 + 144LL * *(unsigned int *)(a1 + 8));
    if ( *(__int64 **)a1 != v14 )
    {
      do
      {
        v14 -= 18;
        v17 = v14[14];
        if ( (__int64 *)v17 != v14 + 16 )
          _libc_free(v17);
        v18 = v14[10];
        if ( (__int64 *)v18 != v14 + 12 )
          _libc_free(v18);
        v19 = v14[6];
        if ( (__int64 *)v19 != v14 + 8 )
          _libc_free(v19);
        v20 = v14[2];
        if ( (__int64 *)v20 != v14 + 4 )
          _libc_free(v20);
      }
      while ( v14 != v16 );
      v14 = *(__int64 **)a1;
    }
  }
  v21 = v24[0];
  if ( v6 != v14 )
    _libc_free((unsigned __int64)v14);
  *(_DWORD *)(a1 + 12) = v21;
  *(_QWORD *)a1 = v23;
  return v23;
}
