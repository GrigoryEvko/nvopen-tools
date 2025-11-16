// Function: sub_1696B90
// Address: 0x1696b90
//
__int64 *__fastcall sub_1696B90(__int64 *a1, __int64 a2, const void *a3, size_t a4)
{
  unsigned int v8; // ebx
  __int64 *v9; // r9
  __int64 v10; // rax
  __int64 *v11; // r9
  __int64 v12; // rcx
  void *v13; // rdi
  __int64 **v14; // rax
  __int64 *v15; // rbx
  __int64 **v16; // rax
  __m128i *v17; // rsi
  __int64 v18; // rax
  void *v19; // rax
  __int64 *v20; // [rsp+0h] [rbp-100h]
  __int64 v21; // [rsp+0h] [rbp-100h]
  __int64 v22; // [rsp+8h] [rbp-F8h]
  __int64 *v23; // [rsp+8h] [rbp-F8h]
  __int64 v24; // [rsp+10h] [rbp-F0h]
  __int64 *v25; // [rsp+10h] [rbp-F0h]
  __int64 v27; // [rsp+20h] [rbp-E0h] BYREF
  __m128i v28; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v29; // [rsp+40h] [rbp-C0h]

  if ( !a4 )
  {
    sub_1693CB0(a1, 9);
    return a1;
  }
  v8 = sub_16D19C0(a2 + 24, a3, a4);
  v9 = (__int64 *)(*(_QWORD *)(a2 + 24) + 8LL * v8);
  if ( *v9 )
  {
    if ( *v9 != -8 )
    {
      *a1 = 1;
      return a1;
    }
    --*(_DWORD *)(a2 + 40);
  }
  v20 = v9;
  v10 = malloc(a4 + 17);
  v11 = v20;
  v12 = v10;
  if ( v10 )
  {
LABEL_9:
    v13 = (void *)(v12 + 16);
    if ( a4 + 1 <= 1 )
      goto LABEL_10;
    goto LABEL_22;
  }
  if ( a4 != -17 || (v18 = malloc(1u), v11 = v20, v12 = 0, !v18) )
  {
    v21 = v12;
    v23 = v11;
    sub_16BD1C0("Allocation failed");
    v11 = v23;
    v12 = v21;
    goto LABEL_9;
  }
  v13 = (void *)(v18 + 16);
  v12 = v18;
LABEL_22:
  v22 = v12;
  v25 = v11;
  v19 = memcpy(v13, a3, a4);
  v12 = v22;
  v11 = v25;
  v13 = v19;
LABEL_10:
  *((_BYTE *)v13 + a4) = 0;
  *(_QWORD *)v12 = a4;
  *(_BYTE *)(v12 + 8) = 0;
  *v11 = v12;
  ++*(_DWORD *)(a2 + 36);
  v14 = (__int64 **)(*(_QWORD *)(a2 + 24) + 8LL * (unsigned int)sub_16D1CD0(a2 + 24, v8));
  v15 = *v14;
  if ( !*v14 || v15 == (__int64 *)-8LL )
  {
    v16 = v14 + 1;
    do
    {
      do
        v15 = *v16++;
      while ( v15 == (__int64 *)-8LL );
    }
    while ( !v15 );
  }
  v24 = *v15;
  sub_16C1840(&v28);
  sub_16C1A90(&v28, a3, a4);
  sub_16C1AA0(&v28, &v27);
  v28.m128i_i64[1] = (__int64)(v15 + 2);
  v17 = *(__m128i **)(a2 + 64);
  v28.m128i_i64[0] = v27;
  v29 = v24;
  if ( v17 == *(__m128i **)(a2 + 72) )
  {
    sub_16969E0((const __m128i **)(a2 + 56), v17, &v28);
  }
  else
  {
    if ( v17 )
    {
      *v17 = _mm_loadu_si128(&v28);
      v17[1].m128i_i64[0] = v29;
      v17 = *(__m128i **)(a2 + 64);
    }
    *(_QWORD *)(a2 + 64) = (char *)v17 + 24;
  }
  *(_BYTE *)(a2 + 128) = 0;
  *a1 = 1;
  return a1;
}
