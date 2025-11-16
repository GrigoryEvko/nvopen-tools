// Function: sub_39C8D10
// Address: 0x39c8d10
//
void __fastcall sub_39C8D10(__int64 a1, _BYTE *a2, __int64 a3, unsigned __int8 *a4)
{
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdi
  __m128i *v9; // rax
  __m128i *v10; // rcx
  __m128i *v11; // rdx
  __m128i *v12; // r15
  unsigned int v13; // r8d
  _QWORD *v14; // r10
  __int64 v15; // rax
  unsigned int v16; // r8d
  _QWORD *v17; // r10
  _QWORD *v18; // rcx
  _BYTE *v19; // rdi
  __int64 v20; // rax
  _BYTE *v21; // rax
  _QWORD *v22; // [rsp+0h] [rbp-D0h]
  _QWORD *v23; // [rsp+0h] [rbp-D0h]
  unsigned int v24; // [rsp+8h] [rbp-C8h]
  _QWORD *v25; // [rsp+8h] [rbp-C8h]
  _QWORD *v26; // [rsp+8h] [rbp-C8h]
  _QWORD *v27; // [rsp+10h] [rbp-C0h]
  unsigned int v28; // [rsp+10h] [rbp-C0h]
  unsigned int v29; // [rsp+18h] [rbp-B8h]
  __int64 v30; // [rsp+28h] [rbp-A8h]
  __m128i *src; // [rsp+30h] [rbp-A0h]
  size_t n; // [rsp+38h] [rbp-98h]
  __m128i v33; // [rsp+40h] [rbp-90h] BYREF
  __m128i v34; // [rsp+50h] [rbp-80h] BYREF
  _QWORD v35[2]; // [rsp+60h] [rbp-70h] BYREF
  __m128i *v36; // [rsp+70h] [rbp-60h] BYREF
  size_t v37; // [rsp+78h] [rbp-58h]
  __m128i v38; // [rsp+80h] [rbp-50h] BYREF
  __int64 v39; // [rsp+90h] [rbp-40h]

  if ( (unsigned __int8)sub_39C8520((_QWORD *)a1) )
  {
    if ( a2 )
    {
      v36 = &v38;
      sub_39C7500((__int64 *)&v36, a2, (__int64)&a2[a3]);
    }
    else
    {
      v37 = 0;
      v36 = &v38;
      v38.m128i_i8[0] = 0;
    }
    sub_39A2AF0(&v34, a1, a4);
    v7 = 15;
    v8 = 15;
    if ( (_QWORD *)v34.m128i_i64[0] != v35 )
      v8 = v35[0];
    if ( v34.m128i_i64[1] + v37 <= v8 )
      goto LABEL_10;
    if ( v36 != &v38 )
      v7 = v38.m128i_i64[0];
    if ( v34.m128i_i64[1] + v37 <= v7 )
    {
      v9 = (__m128i *)sub_2241130((unsigned __int64 *)&v36, 0, 0, v34.m128i_i64[0], v34.m128i_u64[1]);
      src = &v33;
      v10 = (__m128i *)v9->m128i_i64[0];
      v11 = v9 + 1;
      if ( (__m128i *)v9->m128i_i64[0] != &v9[1] )
        goto LABEL_11;
    }
    else
    {
LABEL_10:
      v9 = (__m128i *)sub_2241490((unsigned __int64 *)&v34, v36->m128i_i8, v37);
      src = &v33;
      v10 = (__m128i *)v9->m128i_i64[0];
      v11 = v9 + 1;
      if ( (__m128i *)v9->m128i_i64[0] != &v9[1] )
      {
LABEL_11:
        src = v10;
        v33.m128i_i64[0] = v9[1].m128i_i64[0];
        goto LABEL_12;
      }
    }
    v33 = _mm_loadu_si128(v9 + 1);
LABEL_12:
    n = v9->m128i_u64[1];
    v9->m128i_i64[0] = (__int64)v11;
    v9->m128i_i64[1] = 0;
    v9[1].m128i_i8[0] = 0;
    if ( (_QWORD *)v34.m128i_i64[0] != v35 )
      j_j___libc_free_0(v34.m128i_u64[0]);
    if ( v36 != &v38 )
      j_j___libc_free_0((unsigned __int64)v36);
    v12 = src;
    v36 = &v38;
    v30 = a1 + 672;
    if ( src == &v33 )
    {
      v12 = &v38;
      v38 = _mm_load_si128(&v33);
    }
    else
    {
      v36 = src;
      v38.m128i_i64[0] = v33.m128i_i64[0];
    }
    v33.m128i_i8[0] = 0;
    v37 = n;
    v39 = a1 + 8;
    v13 = sub_16D19C0(v30, (unsigned __int8 *)v12, n);
    v14 = (_QWORD *)(*(_QWORD *)(a1 + 672) + 8LL * v13);
    if ( *v14 )
    {
      if ( *v14 != -8 )
        goto LABEL_20;
      --*(_DWORD *)(a1 + 688);
    }
    v22 = v14;
    v24 = v13;
    v15 = malloc(n + 17);
    v16 = v24;
    v17 = v22;
    v18 = (_QWORD *)v15;
    if ( !v15 )
    {
      if ( n == -17 )
      {
        v20 = malloc(1u);
        v16 = v24;
        v17 = v22;
        v18 = 0;
        if ( v20 )
        {
          v19 = (_BYTE *)(v20 + 16);
          v18 = (_QWORD *)v20;
          goto LABEL_30;
        }
      }
      v23 = v18;
      v26 = v17;
      v28 = v16;
      sub_16BD1C0("Allocation failed", 1u);
      v16 = v28;
      v17 = v26;
      v18 = v23;
    }
    v19 = v18 + 2;
    if ( n + 1 <= 1 )
    {
LABEL_26:
      v19[n] = 0;
      *v18 = n;
      v18[1] = a1 + 8;
      *v17 = v18;
      ++*(_DWORD *)(a1 + 684);
      sub_16D1CD0(v30, v16);
LABEL_20:
      if ( v36 != &v38 )
        j_j___libc_free_0((unsigned __int64)v36);
      return;
    }
LABEL_30:
    v25 = v18;
    v27 = v17;
    v29 = v16;
    v21 = memcpy(v19, v12, n);
    v18 = v25;
    v17 = v27;
    v16 = v29;
    v19 = v21;
    goto LABEL_26;
  }
}
