// Function: sub_37F9C30
// Address: 0x37f9c30
//
_QWORD *__fastcall sub_37F9C30(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r14
  _QWORD *v8; // r13
  _DWORD *v9; // rbx
  __int64 v10; // r12
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rdi
  size_t v18; // rdx
  size_t v19; // r12
  const void *v20; // r13
  unsigned __int64 v21; // rdx
  __m128i *v23; // r13
  size_t v24; // r8
  _QWORD *v25; // rax
  __m128i *v26; // rax
  unsigned __int64 *v27; // rax
  size_t v28; // r8
  __int64 v29; // rdi
  void *v30; // r11
  __m128i *v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rax
  _QWORD *v34; // rdi
  size_t v35; // [rsp+10h] [rbp-C0h]
  size_t v36; // [rsp+18h] [rbp-B8h]
  void *v37; // [rsp+18h] [rbp-B8h]
  __int64 v39; // [rsp+28h] [rbp-A8h]
  _DWORD *v40; // [rsp+30h] [rbp-A0h]
  _DWORD *v41; // [rsp+38h] [rbp-98h]
  unsigned __int64 v42[2]; // [rsp+40h] [rbp-90h] BYREF
  _QWORD v43[2]; // [rsp+50h] [rbp-80h] BYREF
  __m128i *v44; // [rsp+60h] [rbp-70h] BYREF
  __int64 v45; // [rsp+68h] [rbp-68h]
  __m128i v46; // [rsp+70h] [rbp-60h] BYREF
  void *src; // [rsp+80h] [rbp-50h]
  size_t n; // [rsp+88h] [rbp-48h]
  __m128i v49; // [rsp+90h] [rbp-40h] BYREF

  v6 = 0;
  v7 = a2 + 24;
  v8 = (_QWORD *)a2;
  v9 = *(_DWORD **)(a4 + 8);
  v10 = *(_QWORD *)(a4 + 16);
  *(_QWORD *)(a2 + 32) = 0;
  v11 = (v10 - (__int64)v9) >> 2;
  if ( !*(_QWORD *)(a2 + 40) )
  {
    sub_C8D290(v7, (const void *)(a2 + 48), 1, 1u, a5, a6);
    v6 = *(_QWORD *)(a2 + 32);
  }
  *(_BYTE *)(*(_QWORD *)(a2 + 24) + v6) = 40;
  v12 = *(_QWORD *)(a2 + 32) + 1LL;
  *(_QWORD *)(a2 + 32) = v12;
  if ( (_DWORD)v11 )
  {
    v41 = (_DWORD *)(a2 + 16);
    v14 = (unsigned int)(v11 - 1);
    v39 = (__int64)&v9[v14 + 1];
    v40 = &v9[v14];
    while ( 1 )
    {
      v15 = (unsigned int)*v9;
      if ( (unsigned int)v15 >= *v41 )
        break;
      v16 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 8) + 40LL))(*(_QWORD *)(a2 + 8));
      v17 = *(_QWORD *)(a2 + 32);
      v19 = v18;
      v20 = (const void *)v16;
      v21 = v17 + v18;
      if ( v21 > *(_QWORD *)(a2 + 40) )
      {
        sub_C8D290(v7, (const void *)(a2 + 48), v21, 1u, a5, a6);
        v17 = *(_QWORD *)(a2 + 32);
      }
      if ( v19 )
      {
        memcpy((void *)(*(_QWORD *)(a2 + 24) + v17), v20, v19);
        v17 = *(_QWORD *)(a2 + 32);
      }
      v12 = v19 + v17;
      *(_QWORD *)(a2 + 32) = v19 + v17;
LABEL_14:
      if ( v40 != v9 )
      {
        if ( *(_QWORD *)(a2 + 40) < (unsigned __int64)(v12 + 2) )
        {
          sub_C8D290(v7, (const void *)(a2 + 48), v12 + 2, 1u, a5, a6);
          v12 = *(_QWORD *)(a2 + 32);
        }
        ++v9;
        *(_WORD *)(*(_QWORD *)(a2 + 24) + v12) = 8236;
        v12 = *(_QWORD *)(a2 + 32) + 2LL;
        *(_QWORD *)(a2 + 32) = v12;
        if ( (_DWORD *)v39 != v9 )
          continue;
      }
      v8 = (_QWORD *)a2;
      goto LABEL_16;
    }
    if ( !*v9 )
    {
      v49.m128i_i8[0] = 48;
      v23 = &v49;
      v42[0] = (unsigned __int64)v43;
LABEL_21:
      v24 = 1;
      LOBYTE(v43[0]) = v23->m128i_i8[0];
      v25 = v43;
LABEL_22:
      v42[1] = v24;
      *((_BYTE *)v25 + v24) = 0;
      v26 = (__m128i *)sub_2241130(v42, 0, 0, "<unknown 0x", 0xBu);
      v44 = &v46;
      if ( (__m128i *)v26->m128i_i64[0] == &v26[1] )
      {
        v46 = _mm_loadu_si128(v26 + 1);
      }
      else
      {
        v44 = (__m128i *)v26->m128i_i64[0];
        v46.m128i_i64[0] = v26[1].m128i_i64[0];
      }
      v45 = v26->m128i_i64[1];
      v26->m128i_i64[0] = (__int64)v26[1].m128i_i64;
      v26->m128i_i64[1] = 0;
      v26[1].m128i_i8[0] = 0;
      if ( v45 == 0x3FFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"basic_string::append");
      v27 = sub_2241490((unsigned __int64 *)&v44, ">", 1u);
      src = &v49;
      if ( (unsigned __int64 *)*v27 == v27 + 2 )
      {
        v49 = _mm_loadu_si128((const __m128i *)v27 + 1);
      }
      else
      {
        src = (void *)*v27;
        v49.m128i_i64[0] = v27[2];
      }
      n = v27[1];
      *v27 = (unsigned __int64)(v27 + 2);
      v27[1] = 0;
      *((_BYTE *)v27 + 16) = 0;
      v28 = n;
      v29 = *(_QWORD *)(a2 + 32);
      v30 = src;
      if ( n + v29 > *(_QWORD *)(a2 + 40) )
      {
        v35 = n;
        v37 = src;
        sub_C8D290(v7, (const void *)(a2 + 48), n + v29, 1u, n, a6);
        v29 = *(_QWORD *)(a2 + 32);
        v28 = v35;
        v30 = v37;
      }
      if ( v28 )
      {
        v36 = v28;
        memcpy((void *)(*(_QWORD *)(a2 + 24) + v29), v30, v28);
        v29 = *(_QWORD *)(a2 + 32);
        v28 = v36;
      }
      a5 = v29 + v28;
      v31 = (__m128i *)src;
      *(_QWORD *)(a2 + 32) = a5;
      if ( v31 != &v49 )
        j_j___libc_free_0((unsigned __int64)v31);
      if ( v44 != &v46 )
        j_j___libc_free_0((unsigned __int64)v44);
      if ( (_QWORD *)v42[0] != v43 )
        j_j___libc_free_0(v42[0]);
      v12 = *(_QWORD *)(a2 + 32);
      goto LABEL_14;
    }
    v23 = (__m128i *)&v49.m128i_i8[1];
    do
    {
      v23 = (__m128i *)((char *)v23 - 1);
      v32 = v15 & 0xF;
      v15 >>= 4;
      v23->m128i_i8[0] = a0123456789abcd_10[v32];
    }
    while ( v15 );
    v24 = &v49.m128i_i8[1] - (__int8 *)v23;
    v42[0] = (unsigned __int64)v43;
    v44 = (__m128i *)(&v49.m128i_i8[1] - (__int8 *)v23);
    if ( (unsigned __int64)(&v49.m128i_i8[1] - (__int8 *)v23) <= 0xF )
    {
      if ( v24 == 1 )
        goto LABEL_21;
      if ( !v24 )
      {
        v25 = v43;
        goto LABEL_22;
      }
      v34 = v43;
    }
    else
    {
      v33 = sub_22409D0((__int64)v42, (unsigned __int64 *)&v44, 0);
      v24 = &v49.m128i_i8[1] - (__int8 *)v23;
      v42[0] = v33;
      v34 = (_QWORD *)v33;
      v43[0] = v44;
    }
    memcpy(v34, v23, v24);
    v24 = (size_t)v44;
    v25 = (_QWORD *)v42[0];
    goto LABEL_22;
  }
LABEL_16:
  if ( (unsigned __int64)(v12 + 1) > v8[5] )
  {
    sub_C8D290(v7, v8 + 6, v12 + 1, 1u, a5, a6);
    v12 = v8[4];
  }
  *(_BYTE *)(v8[3] + v12) = 41;
  ++v8[4];
  *a1 = 1;
  return a1;
}
