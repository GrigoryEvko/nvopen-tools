// Function: sub_39C9120
// Address: 0x39c9120
//
void __fastcall sub_39C9120(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  __int64 v5; // rdi
  _BYTE *v6; // rax
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdi
  __m128i *v10; // rax
  __m128i *v11; // rcx
  __m128i *v12; // rdx
  __m128i *v13; // r15
  unsigned int v14; // r8d
  _QWORD *v15; // r10
  __int64 v16; // rax
  unsigned int v17; // r8d
  _QWORD *v18; // r10
  _QWORD *v19; // rcx
  _BYTE *v20; // rdi
  __int64 v21; // rax
  _BYTE *v22; // rax
  _QWORD *v23; // [rsp+0h] [rbp-D0h]
  _QWORD *v24; // [rsp+0h] [rbp-D0h]
  unsigned int v25; // [rsp+8h] [rbp-C8h]
  _QWORD *v26; // [rsp+8h] [rbp-C8h]
  unsigned int v27; // [rsp+8h] [rbp-C8h]
  unsigned int v28; // [rsp+10h] [rbp-C0h]
  _QWORD *v29; // [rsp+10h] [rbp-C0h]
  _QWORD *v30; // [rsp+18h] [rbp-B8h]
  __int64 v31; // [rsp+28h] [rbp-A8h]
  __m128i *src; // [rsp+30h] [rbp-A0h]
  size_t n; // [rsp+38h] [rbp-98h]
  __m128i v34; // [rsp+40h] [rbp-90h] BYREF
  __m128i v35; // [rsp+50h] [rbp-80h] BYREF
  _QWORD v36[2]; // [rsp+60h] [rbp-70h] BYREF
  __m128i *v37; // [rsp+70h] [rbp-60h] BYREF
  size_t v38; // [rsp+78h] [rbp-58h]
  __m128i v39; // [rsp+80h] [rbp-50h] BYREF
  __int64 v40; // [rsp+90h] [rbp-40h]

  if ( (unsigned __int8)sub_39C8520((_QWORD *)a1) )
  {
    v5 = *(_QWORD *)(a2 + 8 * (2LL - *(unsigned int *)(a2 + 8)));
    if ( v5 && (v6 = (_BYTE *)sub_161E970(v5)) != 0 )
    {
      v37 = &v39;
      sub_39C7500((__int64 *)&v37, v6, (__int64)&v6[v7]);
    }
    else
    {
      v38 = 0;
      v37 = &v39;
      v39.m128i_i8[0] = 0;
    }
    sub_39A2AF0(&v35, a1, a3);
    v8 = 15;
    v9 = 15;
    if ( (_QWORD *)v35.m128i_i64[0] != v36 )
      v9 = v36[0];
    if ( v35.m128i_i64[1] + v38 <= v9 )
      goto LABEL_11;
    if ( v37 != &v39 )
      v8 = v39.m128i_i64[0];
    if ( v35.m128i_i64[1] + v38 <= v8 )
    {
      v10 = (__m128i *)sub_2241130((unsigned __int64 *)&v37, 0, 0, v35.m128i_i64[0], v35.m128i_u64[1]);
      src = &v34;
      v11 = (__m128i *)v10->m128i_i64[0];
      v12 = v10 + 1;
      if ( (__m128i *)v10->m128i_i64[0] != &v10[1] )
        goto LABEL_12;
    }
    else
    {
LABEL_11:
      v10 = (__m128i *)sub_2241490((unsigned __int64 *)&v35, v37->m128i_i8, v38);
      src = &v34;
      v11 = (__m128i *)v10->m128i_i64[0];
      v12 = v10 + 1;
      if ( (__m128i *)v10->m128i_i64[0] != &v10[1] )
      {
LABEL_12:
        src = v11;
        v34.m128i_i64[0] = v10[1].m128i_i64[0];
        goto LABEL_13;
      }
    }
    v34 = _mm_loadu_si128(v10 + 1);
LABEL_13:
    n = v10->m128i_u64[1];
    v10->m128i_i64[0] = (__int64)v12;
    v10->m128i_i64[1] = 0;
    v10[1].m128i_i8[0] = 0;
    if ( (_QWORD *)v35.m128i_i64[0] != v36 )
      j_j___libc_free_0(v35.m128i_u64[0]);
    if ( v37 != &v39 )
      j_j___libc_free_0((unsigned __int64)v37);
    v13 = src;
    v37 = &v39;
    v31 = a1 + 704;
    if ( src == &v34 )
    {
      v13 = &v39;
      v39 = _mm_load_si128(&v34);
    }
    else
    {
      v37 = src;
      v39.m128i_i64[0] = v34.m128i_i64[0];
    }
    v34.m128i_i8[0] = 0;
    v38 = n;
    v40 = a1 + 8;
    v14 = sub_16D19C0(v31, (unsigned __int8 *)v13, n);
    v15 = (_QWORD *)(*(_QWORD *)(a1 + 704) + 8LL * v14);
    if ( *v15 )
    {
      if ( *v15 != -8 )
        goto LABEL_21;
      --*(_DWORD *)(a1 + 720);
    }
    v23 = v15;
    v25 = v14;
    v16 = malloc(n + 17);
    v17 = v25;
    v18 = v23;
    v19 = (_QWORD *)v16;
    if ( !v16 )
    {
      if ( n == -17 )
      {
        v21 = malloc(1u);
        v19 = 0;
        v17 = v25;
        v18 = v23;
        if ( v21 )
        {
          v20 = (_BYTE *)(v21 + 16);
          v19 = (_QWORD *)v21;
          goto LABEL_31;
        }
      }
      v24 = v18;
      v27 = v17;
      v29 = v19;
      sub_16BD1C0("Allocation failed", 1u);
      v19 = v29;
      v17 = v27;
      v18 = v24;
    }
    v20 = v19 + 2;
    if ( n + 1 <= 1 )
    {
LABEL_27:
      v20[n] = 0;
      *v19 = n;
      v19[1] = a1 + 8;
      *v18 = v19;
      ++*(_DWORD *)(a1 + 716);
      sub_16D1CD0(v31, v17);
LABEL_21:
      if ( v37 != &v39 )
        j_j___libc_free_0((unsigned __int64)v37);
      return;
    }
LABEL_31:
    v26 = v18;
    v28 = v17;
    v30 = v19;
    v22 = memcpy(v20, v13, n);
    v18 = v26;
    v17 = v28;
    v19 = v30;
    v20 = v22;
    goto LABEL_27;
  }
}
