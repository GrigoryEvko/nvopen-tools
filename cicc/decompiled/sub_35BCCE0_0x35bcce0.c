// Function: sub_35BCCE0
// Address: 0x35bcce0
//
unsigned __int64 __fastcall sub_35BCCE0(unsigned __int64 *a1, const __m128i *a2, __int64 *a3)
{
  unsigned __int64 *v3; // rcx
  const __m128i *v5; // r15
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  bool v10; // cf
  unsigned __int64 v11; // rax
  __int8 *v12; // rsi
  __int64 v13; // r13
  __int64 v14; // r8
  char *v15; // rax
  __int64 v16; // rdx
  __m128i v17; // xmm2
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // r13
  unsigned __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  volatile signed __int32 *v25; // rdi
  signed __int32 v26; // eax
  signed __int32 v27; // eax
  const __m128i *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdi
  __m128i v31; // xmm0
  __int64 v32; // rdi
  unsigned __int64 v34; // r13
  __int64 v35; // rax
  unsigned __int64 v36; // [rsp+0h] [rbp-50h]
  unsigned __int64 *v37; // [rsp+8h] [rbp-48h]
  __int64 v38; // [rsp+10h] [rbp-40h]
  __int64 v39; // [rsp+10h] [rbp-40h]
  unsigned __int64 *v40; // [rsp+18h] [rbp-38h]

  v3 = a1;
  v5 = (const __m128i *)a1[1];
  v6 = *a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v5->m128i_i64 - *a1) >> 4);
  if ( v7 == 0x2AAAAAAAAAAAAAALL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  if ( v7 )
    v8 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v5->m128i_i64 - v6) >> 4);
  v10 = __CFADD__(v8, v7);
  v11 = v8 - 0x5555555555555555LL * ((__int64)((__int64)v5->m128i_i64 - v6) >> 4);
  v12 = &a2->m128i_i8[-v6];
  if ( v10 )
  {
    v34 = 0x7FFFFFFFFFFFFFE0LL;
  }
  else
  {
    if ( !v11 )
    {
      v36 = 0;
      v13 = 48;
      v14 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x2AAAAAAAAAAAAAALL )
      v11 = 0x2AAAAAAAAAAAAAALL;
    v34 = 48 * v11;
  }
  v35 = sub_22077B0(v34);
  v3 = a1;
  v14 = v35;
  v36 = v34 + v35;
  v13 = v35 + 48;
LABEL_7:
  v15 = &v12[v14];
  if ( &v12[v14] )
  {
    v16 = *a3;
    v17 = _mm_loadu_si128((const __m128i *)a3 + 2);
    *a3 = 0;
    *(_QWORD *)v15 = v16;
    v18 = a3[1];
    a3[1] = 0;
    *((_QWORD *)v15 + 1) = v18;
    v19 = *(__int64 *)((char *)a3 + 20);
    *((__m128i *)v15 + 2) = v17;
    *(_QWORD *)(v15 + 20) = v19;
  }
  if ( a2 != (const __m128i *)v6 )
  {
    v20 = v14;
    v21 = v6;
    while ( 1 )
    {
      if ( v20 )
      {
        *(_QWORD *)v20 = *(_QWORD *)v21;
        v23 = *(_QWORD *)(v21 + 8);
        *(_QWORD *)(v21 + 8) = 0;
        *(_QWORD *)(v20 + 8) = v23;
        v24 = *(_QWORD *)(v21 + 20);
        *(_QWORD *)v21 = 0;
        *(_QWORD *)(v20 + 20) = v24;
        *(__m128i *)(v20 + 32) = _mm_loadu_si128((const __m128i *)(v21 + 32));
      }
      v25 = *(volatile signed __int32 **)(v21 + 8);
      if ( !v25 )
        goto LABEL_11;
      if ( &_pthread_key_create )
      {
        v26 = _InterlockedExchangeAdd(v25 + 2, 0xFFFFFFFF);
      }
      else
      {
        v26 = *((_DWORD *)v25 + 2);
        *((_DWORD *)v25 + 2) = v26 - 1;
      }
      if ( v26 == 1
        && ((v37 = v3,
             v38 = v14,
             (*(void (**)(void))(*(_QWORD *)v25 + 16LL))(),
             v14 = v38,
             v3 = v37,
             &_pthread_key_create)
          ? (v27 = _InterlockedExchangeAdd(v25 + 3, 0xFFFFFFFF))
          : (v27 = *((_DWORD *)v25 + 3), *((_DWORD *)v25 + 3) = v27 - 1),
            v27 == 1) )
      {
        v21 += 48LL;
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v25 + 24LL))(v25);
        v3 = v37;
        v14 = v38;
        v22 = v20 + 48;
        if ( (const __m128i *)v21 == a2 )
        {
LABEL_23:
          v13 = v20 + 96;
          break;
        }
      }
      else
      {
LABEL_11:
        v21 += 48LL;
        v22 = v20 + 48;
        if ( (const __m128i *)v21 == a2 )
          goto LABEL_23;
      }
      v20 = v22;
    }
  }
  if ( a2 != v5 )
  {
    v28 = a2;
    v29 = v13;
    do
    {
      v30 = v28->m128i_i64[0];
      v31 = _mm_loadu_si128(v28 + 2);
      v28 += 3;
      v29 += 48;
      *(_QWORD *)(v29 - 48) = v30;
      v32 = v28[-3].m128i_i64[1];
      *(__m128i *)(v29 - 16) = v31;
      *(_QWORD *)(v29 - 40) = v32;
      *(_QWORD *)(v29 - 28) = *(__int64 *)((char *)v28[-2].m128i_i64 + 4);
    }
    while ( v28 != v5 );
    v13 += 16
         * (3 * ((0xAAAAAAAAAAAAAABLL * ((unsigned __int64)((char *)v28 - (char *)a2 - 48) >> 4)) & 0xFFFFFFFFFFFFFFFLL)
          + 3);
  }
  if ( v6 )
  {
    v39 = v14;
    v40 = v3;
    j_j___libc_free_0(v6);
    v14 = v39;
    v3 = v40;
  }
  *v3 = v14;
  v3[1] = v13;
  v3[2] = v36;
  return v36;
}
