// Function: sub_3736CA0
// Address: 0x3736ca0
//
void __fastcall sub_3736CA0(__int64 a1, _BYTE *a2, __int64 a3, unsigned __int8 *a4)
{
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdi
  __m128i *v12; // rax
  __m128i *v13; // rcx
  __m128i *v14; // rdx
  __m128i *v15; // r9
  int v16; // eax
  unsigned int v17; // r10d
  _QWORD *v18; // r11
  __int64 v19; // rax
  unsigned int v20; // r10d
  _QWORD *v21; // r11
  _QWORD *v22; // rcx
  __m128i *v23; // rsi
  _QWORD *v24; // [rsp+0h] [rbp-C0h]
  _QWORD *v25; // [rsp+0h] [rbp-C0h]
  unsigned int v26; // [rsp+8h] [rbp-B8h]
  _QWORD *v27; // [rsp+8h] [rbp-B8h]
  __m128i *src; // [rsp+10h] [rbp-B0h]
  unsigned int srca; // [rsp+10h] [rbp-B0h]
  __m128i *v30; // [rsp+20h] [rbp-A0h]
  __int64 n; // [rsp+28h] [rbp-98h]
  __m128i v32; // [rsp+30h] [rbp-90h] BYREF
  __m128i v33; // [rsp+40h] [rbp-80h] BYREF
  _QWORD v34[2]; // [rsp+50h] [rbp-70h] BYREF
  __m128i *v35; // [rsp+60h] [rbp-60h] BYREF
  size_t v36; // [rsp+68h] [rbp-58h]
  __m128i v37; // [rsp+70h] [rbp-50h] BYREF
  __int64 v38; // [rsp+80h] [rbp-40h]

  if ( sub_37365C0((_QWORD *)a1) )
  {
    if ( a2 )
    {
      v35 = &v37;
      sub_3735130((__int64 *)&v35, a2, (__int64)&a2[a3]);
    }
    else
    {
      v36 = 0;
      v35 = &v37;
      v37.m128i_i8[0] = 0;
    }
    sub_3248550(&v33, (char *)a1, a4, v7, v8, v9);
    v10 = 15;
    v11 = 15;
    if ( (_QWORD *)v33.m128i_i64[0] != v34 )
      v11 = v34[0];
    if ( v33.m128i_i64[1] + v36 <= v11 )
      goto LABEL_10;
    if ( v35 != &v37 )
      v10 = v37.m128i_i64[0];
    if ( v33.m128i_i64[1] + v36 <= v10 )
    {
      v12 = (__m128i *)sub_2241130((unsigned __int64 *)&v35, 0, 0, v33.m128i_i64[0], v33.m128i_u64[1]);
      v30 = &v32;
      v13 = (__m128i *)v12->m128i_i64[0];
      v14 = v12 + 1;
      if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
        goto LABEL_11;
    }
    else
    {
LABEL_10:
      v12 = (__m128i *)sub_2241490((unsigned __int64 *)&v33, v35->m128i_i8, v36);
      v30 = &v32;
      v13 = (__m128i *)v12->m128i_i64[0];
      v14 = v12 + 1;
      if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
      {
LABEL_11:
        v30 = v13;
        v32.m128i_i64[0] = v12[1].m128i_i64[0];
        goto LABEL_12;
      }
    }
    v32 = _mm_loadu_si128(v12 + 1);
LABEL_12:
    n = v12->m128i_i64[1];
    v12->m128i_i64[0] = (__int64)v14;
    v12->m128i_i64[1] = 0;
    v12[1].m128i_i8[0] = 0;
    if ( (_QWORD *)v33.m128i_i64[0] != v34 )
      j_j___libc_free_0(v33.m128i_u64[0]);
    if ( v35 != &v37 )
      j_j___libc_free_0((unsigned __int64)v35);
    v15 = v30;
    v35 = &v37;
    if ( v30 == &v32 )
    {
      v15 = &v37;
      v37 = _mm_load_si128(&v32);
    }
    else
    {
      v35 = v30;
      v37.m128i_i64[0] = v32.m128i_i64[0];
    }
    v33.m128i_i64[0] = (__int64)v15;
    src = v15;
    v36 = n;
    v32.m128i_i8[0] = 0;
    v38 = a1 + 8;
    v33.m128i_i64[1] = n;
    v16 = sub_C92610();
    v17 = sub_C92740(a1 + 424, (const void *)v33.m128i_i64[0], v33.m128i_u64[1], v16);
    v18 = (_QWORD *)(*(_QWORD *)(a1 + 424) + 8LL * v17);
    if ( *v18 )
    {
      if ( *v18 != -8 )
      {
LABEL_20:
        if ( v35 != &v37 )
          j_j___libc_free_0((unsigned __int64)v35);
        return;
      }
      --*(_DWORD *)(a1 + 440);
    }
    v24 = v18;
    v26 = v17;
    v19 = sub_C7D670(n + 17, 8);
    v20 = v26;
    v21 = v24;
    v22 = (_QWORD *)v19;
    if ( n )
    {
      v23 = src;
      v27 = v24;
      srca = v20;
      v25 = (_QWORD *)v19;
      memcpy((void *)(v19 + 16), v23, n);
      v20 = srca;
      v21 = v27;
      v22 = v25;
    }
    *((_BYTE *)v22 + n + 16) = 0;
    *v22 = n;
    v22[1] = a1 + 8;
    *v21 = v22;
    ++*(_DWORD *)(a1 + 436);
    sub_C929D0((__int64 *)(a1 + 424), v20);
    goto LABEL_20;
  }
}
