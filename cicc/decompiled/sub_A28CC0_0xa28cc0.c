// Function: sub_A28CC0
// Address: 0xa28cc0
//
__int64 __fastcall sub_A28CC0(_QWORD *a1, __m128i *a2)
{
  __int64 v2; // r12
  __int64 v3; // r15
  const void *v4; // r13
  size_t v5; // r14
  size_t v6; // rbx
  size_t v7; // rdx
  int v8; // eax
  size_t v9; // rbx
  size_t v10; // rdx
  int v11; // eax
  __m128i *v13; // [rsp+28h] [rbp-38h] BYREF

  v2 = (__int64)(a1 + 1);
  v3 = a1[2];
  if ( !v3 )
  {
    v2 = (__int64)(a1 + 1);
    goto LABEL_24;
  }
  v4 = (const void *)a2->m128i_i64[0];
  v5 = a2->m128i_u64[1];
  do
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(v3 + 40);
      v7 = v5;
      if ( v6 <= v5 )
        v7 = *(_QWORD *)(v3 + 40);
      if ( v7 )
      {
        v8 = memcmp(*(const void **)(v3 + 32), v4, v7);
        if ( v8 )
          break;
      }
      if ( (__int64)(v6 - v5) >= 0x80000000LL )
        goto LABEL_12;
      if ( (__int64)(v6 - v5) > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v8 = v6 - v5;
        break;
      }
LABEL_3:
      v3 = *(_QWORD *)(v3 + 24);
      if ( !v3 )
        goto LABEL_13;
    }
    if ( v8 < 0 )
      goto LABEL_3;
LABEL_12:
    v2 = v3;
    v3 = *(_QWORD *)(v3 + 16);
  }
  while ( v3 );
LABEL_13:
  if ( a1 + 1 == (_QWORD *)v2 )
    goto LABEL_24;
  v9 = *(_QWORD *)(v2 + 40);
  v10 = v5;
  if ( v9 <= v5 )
    v10 = *(_QWORD *)(v2 + 40);
  if ( v10 && (v11 = memcmp(v4, *(const void **)(v2 + 32), v10)) != 0 )
  {
LABEL_21:
    if ( v11 < 0 )
      goto LABEL_24;
  }
  else if ( (__int64)(v5 - v9) <= 0x7FFFFFFF )
  {
    if ( (__int64)(v5 - v9) >= (__int64)0xFFFFFFFF80000000LL )
    {
      v11 = v5 - v9;
      goto LABEL_21;
    }
LABEL_24:
    v13 = a2;
    v2 = sub_A28B80(a1, (_QWORD *)v2, &v13);
  }
  return v2 + 64;
}
