// Function: sub_2F8DE40
// Address: 0x2f8de40
//
unsigned int *__fastcall sub_2F8DE40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 *v7; // rbx
  __int64 v8; // rdx
  unsigned int *result; // rax
  __int64 *i; // r13
  __int64 *v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // r13
  __int64 v15; // r14
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 *v19; // rbx
  __int64 *v20; // r13
  unsigned __int64 v21; // rdi
  __int64 v22; // [rsp+0h] [rbp-40h] BYREF
  __int64 v23; // [rsp+8h] [rbp-38h]
  __int64 *v24; // [rsp+10h] [rbp-30h]

  v6 = *(unsigned int *)(a1 + 1312);
  if ( (unsigned int)v6 > 2 )
  {
    v12 = *(_QWORD *)(a1 + 1304);
    v13 = 88 * v6;
    v14 = (__int64 *)(v12 + 88);
    v15 = v12 + v13;
    sub_2F8CCE0(&v22, v12 + 88, 0x2E8BA2E8BA2E8BA3LL * ((v13 - 88) >> 3));
    if ( v24 )
      sub_2F8C650(v14, v15, v24, v23, v17, v18);
    else
      sub_2F8BBB0((__int64)v14, v15, 0, v16, v17, v18);
    v19 = v24;
    v20 = &v24[11 * v23];
    if ( v24 != v20 )
    {
      do
      {
        v21 = v19[2];
        if ( (__int64 *)v21 != v19 + 4 )
          _libc_free(v21);
        v19 += 11;
      }
      while ( v20 != v19 );
      v20 = v24;
    }
    j_j___libc_free_0((unsigned __int64)v20);
    v6 = *(unsigned int *)(a1 + 1312);
  }
  v7 = *(__int64 **)(a1 + 1304);
  v8 = 5 * v6;
  result = (unsigned int *)(11 * v6);
  for ( i = &v7[(_QWORD)result]; i != v7; result = sub_2F8CEE0(a1, v11, v8, a4, a5) )
  {
    v11 = v7;
    v7 += 11;
  }
  return result;
}
