// Function: sub_16DB620
// Address: 0x16db620
//
char __fastcall sub_16DB620(__int64 a1)
{
  int v1; // eax
  __int64 v2; // rbx
  __int64 i; // r13
  _QWORD *v4; // r12
  char *v5; // r14
  size_t v6; // r12
  __m128i *v7; // r12
  __m128i *v8; // r12
  __m128i *v10; // [rsp-68h] [rbp-68h] BYREF
  const void *v11; // [rsp-60h] [rbp-60h]
  __int64 v12; // [rsp-58h] [rbp-58h]
  __m128i *v13; // [rsp-48h] [rbp-48h] BYREF
  const void *v14; // [rsp-40h] [rbp-40h]
  __int64 v15; // [rsp-38h] [rbp-38h]

  v1 = *(_DWORD *)(a1 + 24);
  if ( !v1 )
    return v1;
  sub_16DA620(&v10, (__m128i *)0xFFFFFFFFFFFFFFFFLL, 0);
  LOBYTE(v1) = sub_16DA620(&v13, (__m128i *)0xFFFFFFFFFFFFFFFELL, 0);
  v2 = *(_QWORD *)(a1 + 8);
  for ( i = v2 + ((unsigned __int64)*(unsigned int *)(a1 + 24) << 6); i != v2; v2 += 64 )
  {
    v5 = *(char **)(v2 + 8);
    v6 = *(_QWORD *)(v2 + 16);
    LOBYTE(v1) = v5 + 1 == 0;
    if ( v11 == (const void *)-1LL || (LOBYTE(v1) = v5 + 2 == 0, v11 == (const void *)-2LL) )
    {
      if ( (_BYTE)v1 )
        goto LABEL_10;
    }
    else if ( v6 == v12 )
    {
      if ( !v6 )
        goto LABEL_10;
      v1 = memcmp(*(const void **)(v2 + 8), v11, *(_QWORD *)(v2 + 16));
      if ( !v1 )
        goto LABEL_10;
    }
    LOBYTE(v1) = v5 + 1 == 0;
    if ( v14 == (const void *)-1LL || (LOBYTE(v1) = v5 + 2 == 0, v14 == (const void *)-2LL) )
    {
      if ( (_BYTE)v1 )
        goto LABEL_10;
    }
    else if ( v6 == v15 )
    {
      if ( !v6 )
        goto LABEL_10;
      v1 = memcmp(v5, v14, v6);
      if ( !v1 )
        goto LABEL_10;
    }
    LOBYTE(v1) = sub_16F2AA0(v2 + 24);
LABEL_10:
    v4 = *(_QWORD **)v2;
    if ( *(_QWORD *)v2 )
    {
      if ( (_QWORD *)*v4 != v4 + 2 )
        j_j___libc_free_0(*v4, v4[2] + 1LL);
      LOBYTE(v1) = j_j___libc_free_0(v4, 32);
    }
  }
  v7 = v13;
  if ( v13 )
  {
    if ( (__m128i *)v13->m128i_i64[0] != &v13[1] )
      j_j___libc_free_0(v13->m128i_i64[0], v13[1].m128i_i64[0] + 1);
    LOBYTE(v1) = j_j___libc_free_0(v7, 32);
  }
  v8 = v10;
  if ( v10 )
  {
    if ( (__m128i *)v10->m128i_i64[0] != &v10[1] )
      j_j___libc_free_0(v10->m128i_i64[0], v10[1].m128i_i64[0] + 1);
    LOBYTE(v1) = j_j___libc_free_0(v8, 32);
  }
  return v1;
}
