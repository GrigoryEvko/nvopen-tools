// Function: sub_39C8580
// Address: 0x39c8580
//
void __fastcall sub_39C8580(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, unsigned __int8 *a5)
{
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdi
  __m128i *v10; // rax
  __m128i *v11; // rcx
  __m128i *v12; // rdx
  unsigned int v13; // r9d
  _QWORD *v14; // r10
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned int v17; // r9d
  _QWORD *v18; // r10
  _QWORD *v19; // rcx
  _BYTE *v20; // rdi
  __int64 *v21; // rax
  __int64 *v22; // rax
  __int64 v23; // rax
  _BYTE *v24; // rax
  _QWORD *v25; // [rsp+8h] [rbp-B8h]
  _QWORD *v26; // [rsp+8h] [rbp-B8h]
  unsigned int v27; // [rsp+10h] [rbp-B0h]
  _QWORD *v28; // [rsp+10h] [rbp-B0h]
  _QWORD *v29; // [rsp+10h] [rbp-B0h]
  _QWORD *v30; // [rsp+18h] [rbp-A8h]
  unsigned int v31; // [rsp+18h] [rbp-A8h]
  unsigned int v32; // [rsp+20h] [rbp-A0h]
  __m128i *src; // [rsp+30h] [rbp-90h]
  size_t n; // [rsp+38h] [rbp-88h]
  __m128i v36; // [rsp+40h] [rbp-80h] BYREF
  __m128i v37; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v38[2]; // [rsp+60h] [rbp-60h] BYREF
  char *v39; // [rsp+70h] [rbp-50h] BYREF
  size_t v40; // [rsp+78h] [rbp-48h]
  _QWORD v41[8]; // [rsp+80h] [rbp-40h] BYREF

  if ( (unsigned __int8)sub_39C8520((_QWORD *)a1) )
  {
    if ( a2 )
    {
      v39 = (char *)v41;
      sub_39C7500((__int64 *)&v39, a2, (__int64)&a2[a3]);
    }
    else
    {
      v40 = 0;
      v39 = (char *)v41;
      LOBYTE(v41[0]) = 0;
    }
    sub_39A2AF0(&v37, a1, a5);
    v8 = 15;
    v9 = 15;
    if ( (_QWORD *)v37.m128i_i64[0] != v38 )
      v9 = v38[0];
    if ( v37.m128i_i64[1] + v40 <= v9 )
      goto LABEL_11;
    if ( v39 != (char *)v41 )
      v8 = v41[0];
    if ( v37.m128i_i64[1] + v40 <= v8 )
    {
      v10 = (__m128i *)sub_2241130((unsigned __int64 *)&v39, 0, 0, v37.m128i_i64[0], v37.m128i_u64[1]);
      v12 = v10 + 1;
      src = &v36;
      v11 = (__m128i *)v10->m128i_i64[0];
      if ( (__m128i *)v10->m128i_i64[0] != &v10[1] )
        goto LABEL_12;
    }
    else
    {
LABEL_11:
      v10 = (__m128i *)sub_2241490((unsigned __int64 *)&v37, v39, v40);
      src = &v36;
      v11 = (__m128i *)v10->m128i_i64[0];
      v12 = v10 + 1;
      if ( (__m128i *)v10->m128i_i64[0] != &v10[1] )
      {
LABEL_12:
        src = v11;
        v36.m128i_i64[0] = v10[1].m128i_i64[0];
        goto LABEL_13;
      }
    }
    v36 = _mm_loadu_si128(v10 + 1);
LABEL_13:
    n = v10->m128i_u64[1];
    v10->m128i_i64[0] = (__int64)v12;
    v10->m128i_i64[1] = 0;
    v10[1].m128i_i8[0] = 0;
    if ( (_QWORD *)v37.m128i_i64[0] != v38 )
      j_j___libc_free_0(v37.m128i_u64[0]);
    if ( v39 != (char *)v41 )
      j_j___libc_free_0((unsigned __int64)v39);
    v13 = sub_16D19C0(a1 + 672, (unsigned __int8 *)src, n);
    v14 = (_QWORD *)(*(_QWORD *)(a1 + 672) + 8LL * v13);
    v15 = *v14;
    if ( *v14 )
    {
      if ( v15 != -8 )
      {
LABEL_19:
        *(_QWORD *)(v15 + 8) = a4;
        if ( src != &v36 )
          j_j___libc_free_0((unsigned __int64)src);
        return;
      }
      --*(_DWORD *)(a1 + 688);
    }
    v25 = v14;
    v27 = v13;
    v16 = malloc(n + 17);
    v17 = v27;
    v18 = v25;
    v19 = (_QWORD *)v16;
    if ( !v16 )
    {
      if ( n == -17 )
      {
        v23 = malloc(1u);
        v17 = v27;
        v18 = v25;
        v19 = 0;
        if ( v23 )
        {
          v20 = (_BYTE *)(v23 + 16);
          v19 = (_QWORD *)v23;
          goto LABEL_33;
        }
      }
      v26 = v19;
      v29 = v18;
      v31 = v17;
      sub_16BD1C0("Allocation failed", 1u);
      v17 = v31;
      v18 = v29;
      v19 = v26;
    }
    v20 = v19 + 2;
    if ( n + 1 <= 1 )
    {
LABEL_24:
      v20[n] = 0;
      *v19 = n;
      v19[1] = 0;
      *v18 = v19;
      ++*(_DWORD *)(a1 + 684);
      v21 = (__int64 *)(*(_QWORD *)(a1 + 672) + 8LL * (unsigned int)sub_16D1CD0(a1 + 672, v17));
      v15 = *v21;
      if ( !*v21 || v15 == -8 )
      {
        v22 = v21 + 1;
        do
        {
          do
            v15 = *v22++;
          while ( !v15 );
        }
        while ( v15 == -8 );
      }
      goto LABEL_19;
    }
LABEL_33:
    v28 = v19;
    v30 = v18;
    v32 = v17;
    v24 = memcpy(v20, src, n);
    v19 = v28;
    v18 = v30;
    v17 = v32;
    v20 = v24;
    goto LABEL_24;
  }
}
