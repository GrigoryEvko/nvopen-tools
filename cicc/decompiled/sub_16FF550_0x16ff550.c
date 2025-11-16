// Function: sub_16FF550
// Address: 0x16ff550
//
unsigned __int64 *__fastcall sub_16FF550(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 *v5; // r13
  __int64 v6; // rax
  __int64 v7; // r14
  unsigned __int64 v8; // r12
  unsigned __int64 *v9; // rbx
  unsigned __int64 *v10; // r14
  unsigned __int64 v11; // rdi
  unsigned __int64 *v12; // rbx
  unsigned __int64 v13; // r14
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  char v17[24]; // [rsp+0h] [rbp-60h] BYREF
  __int64 *v18; // [rsp+18h] [rbp-48h]
  __int64 v19; // [rsp+28h] [rbp-38h] BYREF

  if ( a1[1] )
    sub_16BD130("Can only iterate over the stream once", 1u);
  sub_16FC0B0((__int64)v17, *a1, a3, a4, a5);
  if ( v18 != &v19 )
    j_j___libc_free_0(v18, v19 + 1);
  v5 = a1 + 1;
  v6 = sub_22077B0(168);
  v7 = v6;
  if ( v6 )
    sub_16FF2B0(v6, (__int64)a1);
  v8 = a1[1];
  a1[1] = v7;
  if ( v8 )
  {
    sub_16F67F0(*(_QWORD *)(v8 + 136));
    v9 = *(unsigned __int64 **)(v8 + 24);
    v10 = &v9[*(unsigned int *)(v8 + 32)];
    while ( v10 != v9 )
    {
      v11 = *v9++;
      _libc_free(v11);
    }
    v12 = *(unsigned __int64 **)(v8 + 72);
    v13 = (unsigned __int64)&v12[2 * *(unsigned int *)(v8 + 80)];
    if ( v12 != (unsigned __int64 *)v13 )
    {
      do
      {
        v14 = *v12;
        v12 += 2;
        _libc_free(v14);
      }
      while ( (unsigned __int64 *)v13 != v12 );
      v13 = *(_QWORD *)(v8 + 72);
    }
    if ( v13 != v8 + 88 )
      _libc_free(v13);
    v15 = *(_QWORD *)(v8 + 24);
    if ( v15 != v8 + 40 )
      _libc_free(v15);
    j_j___libc_free_0(v8, 168);
  }
  return v5;
}
