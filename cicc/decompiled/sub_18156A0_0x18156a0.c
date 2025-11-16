// Function: sub_18156A0
// Address: 0x18156a0
//
__int64 __fastcall sub_18156A0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r14
  __int64 v6; // r13
  unsigned __int64 v7; // r15
  __int64 *v8; // rbx
  const __m128i *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rsi
  __int64 v13; // r14
  __int64 v14; // r13
  __int64 result; // rax
  __int64 v16; // r13
  __int64 v17; // r12
  __int64 *v18; // rbx
  __int64 *v19; // r12
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // r14
  bool v22; // cf
  unsigned __int64 v23; // rax
  const __m128i *v24; // rax
  char *v25; // r13
  __m128i *v26; // rdx
  __m128i *v27; // r13
  const __m128i *v28; // rdi
  __int64 v29; // r14
  __int64 *v30; // r15
  const __m128i *v31; // rax
  __m128i *v32; // rbx
  __int64 v33; // rcx
  const __m128i *v34; // rcx
  __int64 *v35; // r13
  __int64 v36; // r14
  __int64 v37; // rax
  __int64 v38; // [rsp+8h] [rbp-78h]
  __int64 v39; // [rsp+10h] [rbp-70h]
  __m128i *v40; // [rsp+18h] [rbp-68h]
  __int64 v41; // [rsp+28h] [rbp-58h] BYREF
  __int64 *v42; // [rsp+30h] [rbp-50h] BYREF
  __int64 *v43; // [rsp+38h] [rbp-48h]
  char *v44; // [rsp+40h] [rbp-40h]

  *(_QWORD *)(a1 + 16) = &unk_4FA8E6C;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  *(_QWORD *)(a1 + 128) = a1 + 112;
  *(_QWORD *)(a1 + 136) = a1 + 112;
  *(_QWORD *)a1 = off_49F0998;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 5;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 240) = a3;
  *(_QWORD *)(a1 + 248) = a4;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 416) = 0;
  *(_DWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 432) = 0;
  *(_DWORD *)(a1 + 448) = 0;
  *(_QWORD *)(a1 + 456) = 0;
  *(_QWORD *)(a1 + 464) = a1 + 448;
  *(_QWORD *)(a1 + 472) = a1 + 448;
  *(_QWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_QWORD *)(a1 + 504) = 0;
  *(_QWORD *)(a1 + 512) = 0;
  *(_QWORD *)(a1 + 520) = 0;
  *(_BYTE *)(a1 + 528) = 0;
  v5 = a2[1];
  v6 = *a2;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v7 = v5 - v6;
  if ( v5 == v6 )
  {
    v7 = 0;
    v8 = 0;
  }
  else
  {
    if ( v7 > 0x7FFFFFFFFFFFFFE0LL )
      sub_4261EA(a1, a2, a3);
    v8 = (__int64 *)sub_22077B0(v5 - v6);
    v5 = a2[1];
    v6 = *a2;
  }
  v42 = v8;
  v43 = v8;
  v44 = (char *)v8 + v7;
  if ( v6 == v5 )
  {
    v9 = (const __m128i *)v8;
    v10 = 0;
  }
  else
  {
    do
    {
      if ( v8 )
      {
        *v8 = (__int64)(v8 + 2);
        sub_1814C60(v8, *(_BYTE **)v6, *(_QWORD *)v6 + *(_QWORD *)(v6 + 8));
      }
      v6 += 32;
      v8 += 4;
    }
    while ( v5 != v6 );
    v9 = (const __m128i *)v42;
    v10 = (char *)v8 - (char *)v42;
  }
  v11 = qword_4FA92A8;
  v12 = qword_4FA92A0;
  v43 = v8;
  if ( qword_4FA92A0 != qword_4FA92A8 )
  {
    v13 = qword_4FA92A8 - qword_4FA92A0;
    if ( v44 - (char *)v8 >= (unsigned __int64)(qword_4FA92A8 - qword_4FA92A0) )
    {
      v14 = qword_4FA92A0;
      do
      {
        if ( v8 )
        {
          *v8 = (__int64)(v8 + 2);
          sub_1814C60(v8, *(_BYTE **)v14, *(_QWORD *)v14 + *(_QWORD *)(v14 + 8));
        }
        v14 += 32;
        v8 += 4;
      }
      while ( v11 != v14 );
      v43 = (__int64 *)((char *)v43 + v13);
      goto LABEL_16;
    }
    v20 = v10 >> 5;
    v21 = v13 >> 5;
    if ( v21 > 0x3FFFFFFFFFFFFFFLL - v20 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v21 < v20 )
      v21 = v20;
    v22 = __CFADD__(v21, v20);
    v23 = v21 + v20;
    if ( v22 )
    {
      v36 = 0x7FFFFFFFFFFFFFE0LL;
    }
    else
    {
      if ( !v23 )
      {
        v39 = 0;
        v40 = 0;
LABEL_33:
        if ( v9 == (const __m128i *)v8 )
        {
          v27 = v40;
        }
        else
        {
          v24 = v9 + 1;
          v25 = (char *)((char *)v8 - (char *)v9);
          v26 = v40;
          v27 = (__m128i *)&v25[(_QWORD)v40];
          do
          {
            if ( v26 )
            {
              v26->m128i_i64[0] = (__int64)v26[1].m128i_i64;
              v28 = (const __m128i *)v24[-1].m128i_i64[0];
              if ( v28 == v24 )
              {
                v26[1] = _mm_loadu_si128(v24);
              }
              else
              {
                v26->m128i_i64[0] = (__int64)v28;
                v26[1].m128i_i64[0] = v24->m128i_i64[0];
              }
              v26->m128i_i64[1] = v24[-1].m128i_i64[1];
              v24[-1].m128i_i64[0] = (__int64)v24;
              v24[-1].m128i_i64[1] = 0;
              v24->m128i_i8[0] = 0;
            }
            v26 += 2;
            v24 += 2;
          }
          while ( v26 != v27 );
        }
        v29 = v12;
        do
        {
          if ( v27 )
          {
            v27->m128i_i64[0] = (__int64)v27[1].m128i_i64;
            sub_1814C60(v27->m128i_i64, *(_BYTE **)v29, *(_QWORD *)v29 + *(_QWORD *)(v29 + 8));
          }
          v29 += 32;
          v27 += 2;
        }
        while ( v11 != v29 );
        v30 = v43;
        if ( v43 == v8 )
        {
          v32 = v27;
        }
        else
        {
          v31 = (const __m128i *)(v8 + 2);
          v32 = (__m128i *)((char *)v27 + (char *)v43 - (char *)v8);
          do
          {
            if ( v27 )
            {
              v27->m128i_i64[0] = (__int64)v27[1].m128i_i64;
              v34 = (const __m128i *)v31[-1].m128i_i64[0];
              if ( v34 == v31 )
              {
                v27[1] = _mm_loadu_si128(v31);
              }
              else
              {
                v27->m128i_i64[0] = (__int64)v34;
                v27[1].m128i_i64[0] = v31->m128i_i64[0];
              }
              v33 = v31[-1].m128i_i64[1];
              v31[-1].m128i_i64[0] = (__int64)v31;
              v31[-1].m128i_i64[1] = 0;
              v27->m128i_i64[1] = v33;
              v31->m128i_i8[0] = 0;
            }
            v27 += 2;
            v31 += 2;
          }
          while ( v27 != v32 );
        }
        v35 = v42;
        if ( v30 != v42 )
        {
          do
          {
            if ( (__int64 *)*v35 != v35 + 2 )
              j_j___libc_free_0(*v35, v35[2] + 1);
            v35 += 4;
          }
          while ( v30 != v35 );
          v35 = v42;
        }
        if ( v35 )
          j_j___libc_free_0(v35, v44 - (char *)v35);
        v43 = (__int64 *)v32;
        v42 = (__int64 *)v40;
        v44 = (char *)v39;
        goto LABEL_16;
      }
      if ( v23 > 0x3FFFFFFFFFFFFFFLL )
        v23 = 0x3FFFFFFFFFFFFFFLL;
      v36 = 32 * v23;
    }
    v38 = qword_4FA92A0;
    v37 = sub_22077B0(v36);
    v9 = (const __m128i *)v42;
    v12 = v38;
    v40 = (__m128i *)v37;
    v39 = v37 + v36;
    goto LABEL_33;
  }
LABEL_16:
  sub_394A370(&v41, &v42);
  result = v41;
  v16 = *(_QWORD *)(a1 + 392);
  v41 = 0;
  *(_QWORD *)(a1 + 392) = result;
  if ( v16 )
  {
    sub_39479B0(v16);
    result = j_j___libc_free_0(v16, 24);
    v17 = v41;
    if ( v41 )
    {
      sub_39479B0(v41);
      result = j_j___libc_free_0(v17, 24);
    }
  }
  v18 = v43;
  v19 = v42;
  if ( v43 != v42 )
  {
    do
    {
      result = (__int64)(v19 + 2);
      if ( (__int64 *)*v19 != v19 + 2 )
        result = j_j___libc_free_0(*v19, v19[2] + 1);
      v19 += 4;
    }
    while ( v18 != v19 );
    v19 = v42;
  }
  if ( v19 )
    return j_j___libc_free_0(v19, v44 - (char *)v19);
  return result;
}
