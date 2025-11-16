// Function: sub_CAFE70
// Address: 0xcafe70
//
unsigned __int64 *__fastcall sub_CAFE70(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rsi
  unsigned __int64 *v6; // r14
  __int64 v7; // rax
  __int64 v8; // r13
  unsigned __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // rdi
  __int64 *v12; // r15
  __int64 *v13; // rbx
  __int64 i; // rax
  __int64 v15; // rdi
  unsigned int v16; // ecx
  __int64 *v17; // rbx
  __int64 *v18; // r13
  __int64 v19; // rdi
  __int64 v20; // rdi
  char v22[24]; // [rsp+0h] [rbp-70h] BYREF
  __int64 *v23; // [rsp+18h] [rbp-58h]
  __int64 v24; // [rsp+28h] [rbp-48h] BYREF

  if ( a1[1] )
    sub_C64ED0("Can only iterate over the stream once", 1u);
  v5 = *a1;
  sub_CAD4E0((__int64)v22, *a1, a3, a4, a5);
  if ( v23 != &v24 )
  {
    v5 = v24 + 1;
    j_j___libc_free_0(v23, v24 + 1);
  }
  v6 = a1 + 1;
  v7 = sub_22077B0(160);
  v8 = v7;
  if ( v7 )
  {
    v5 = (__int64)a1;
    sub_CAFBE0(v7, (__int64)a1);
  }
  v9 = a1[1];
  a1[1] = v8;
  if ( v9 )
  {
    v10 = *(_QWORD *)(v9 + 128);
    while ( v10 )
    {
      sub_CA65A0(*(_QWORD *)(v10 + 24));
      v11 = v10;
      v10 = *(_QWORD *)(v10 + 16);
      v5 = 64;
      j_j___libc_free_0(v11, 64);
    }
    v12 = *(__int64 **)(v9 + 24);
    v13 = &v12[*(unsigned int *)(v9 + 32)];
    if ( v12 != v13 )
    {
      for ( i = *(_QWORD *)(v9 + 24); ; i = *(_QWORD *)(v9 + 24) )
      {
        v15 = *v12;
        v16 = (unsigned int)(((__int64)v12 - i) >> 3) >> 7;
        v5 = 4096LL << v16;
        if ( v16 >= 0x1E )
          v5 = 0x40000000000LL;
        ++v12;
        sub_C7D6A0(v15, v5, 16);
        if ( v13 == v12 )
          break;
      }
    }
    v17 = *(__int64 **)(v9 + 72);
    v18 = &v17[2 * *(unsigned int *)(v9 + 80)];
    if ( v17 != v18 )
    {
      do
      {
        v5 = v17[1];
        v19 = *v17;
        v17 += 2;
        sub_C7D6A0(v19, v5, 16);
      }
      while ( v18 != v17 );
      v18 = *(__int64 **)(v9 + 72);
    }
    if ( v18 != (__int64 *)(v9 + 88) )
      _libc_free(v18, v5);
    v20 = *(_QWORD *)(v9 + 24);
    if ( v20 != v9 + 40 )
      _libc_free(v20, v5);
    j_j___libc_free_0(v9, 160);
  }
  return v6;
}
