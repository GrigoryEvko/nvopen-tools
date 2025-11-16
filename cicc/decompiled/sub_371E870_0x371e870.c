// Function: sub_371E870
// Address: 0x371e870
//
__int64 __fastcall sub_371E870(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __m128i *v7; // r15
  __int64 v8; // r12
  __m128i *v10; // r9
  __int64 v11; // r13
  __m128i *v12; // r11
  unsigned int v13; // ecx
  unsigned int v14; // r10d
  bool v15; // dl
  unsigned int v16; // eax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int32 v19; // eax
  __int64 v20; // r10
  unsigned __int64 v21; // r14
  __m128i *v22; // rcx
  unsigned int v23; // esi
  unsigned int v24; // edi
  bool v25; // al
  __m128i *i; // rax
  unsigned int v27; // edx
  bool v28; // di
  __int64 v29; // rdx
  __int32 v30; // eax
  bool v31; // dl
  __int64 v32; // rsi
  __int64 v33; // rcx
  __int32 v34; // edi
  bool v35; // dl
  unsigned __int64 v36; // rdx
  __m128i *v37; // rsi
  __m128i *v38; // r15
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rax
  __int64 v43; // r13
  __m128i v44; // xmm1
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int32 v48; // eax
  __m128i v49; // xmm5
  int v50; // eax
  __int64 v51; // rcx
  __m128i v52; // xmm6
  __int64 v53; // rax

  result = (__int64)a2->m128i_i64 - a1;
  if ( (__int64)a2->m128i_i64 - a1 <= 384 )
    return result;
  v7 = a2;
  v8 = a3;
  if ( !a3 )
    goto LABEL_34;
  v10 = a2;
  v11 = a1 + 24;
  while ( 2 )
  {
    --v8;
    v12 = (__m128i *)(a1
                    + 8
                    * ((__int64)(0xAAAAAAAAAAAAAAABLL * (((__int64)v10->m128i_i64 - a1) >> 3)) / 2
                     + ((0xAAAAAAAAAAAAAAABLL * (((__int64)v10->m128i_i64 - a1) >> 3)
                       + ((0xAAAAAAAAAAAAAAABLL * (((__int64)v10->m128i_i64 - a1) >> 3)) >> 63))
                      & 0xFFFFFFFFFFFFFFFELL)));
    v13 = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 72LL);
    v14 = *(_DWORD *)(v12->m128i_i64[1] + 72);
    v15 = v13 < v14;
    if ( v13 == v14 )
      v15 = *(_DWORD *)(a1 + 40) < v12[1].m128i_i32[0];
    v16 = *(_DWORD *)(v10[-1].m128i_i64[0] + 72);
    if ( !v15 )
    {
      v31 = v13 < v16;
      if ( v13 == v16 )
        v31 = *(_DWORD *)(a1 + 40) < v10[-1].m128i_i32[2];
      v32 = *(_QWORD *)a1;
      v33 = *(_QWORD *)(a1 + 8);
      v34 = *(_DWORD *)(a1 + 16);
      if ( v31 )
      {
        v52 = _mm_loadu_si128((const __m128i *)(a1 + 24));
        v53 = *(_QWORD *)(a1 + 40);
        *(_QWORD *)(a1 + 24) = v32;
        v20 = v33;
        *(_QWORD *)(a1 + 32) = v33;
        *(_QWORD *)(a1 + 16) = v53;
        *(_DWORD *)(a1 + 40) = v34;
        *(__m128i *)a1 = v52;
        v18 = v10[-1].m128i_i64[0];
      }
      else
      {
        v35 = v14 < v16;
        if ( v14 == v16 )
          v35 = v12[1].m128i_i32[0] < (unsigned __int32)v10[-1].m128i_i32[2];
        if ( v35 )
        {
          v18 = *(_QWORD *)(a1 + 8);
          *(__m128i *)a1 = _mm_loadu_si128((__m128i *)((char *)v10 - 24));
          *(_QWORD *)(a1 + 16) = v10[-1].m128i_i64[1];
          v10[-2].m128i_i64[1] = v32;
          v10[-1].m128i_i64[0] = v33;
          v10[-1].m128i_i32[2] = v34;
          v20 = *(_QWORD *)(a1 + 32);
        }
        else
        {
          *(__m128i *)a1 = _mm_loadu_si128(v12);
          *(_QWORD *)(a1 + 16) = v12[1].m128i_i64[0];
          v12->m128i_i64[0] = v32;
          v12->m128i_i64[1] = v33;
          v12[1].m128i_i32[0] = v34;
          v20 = *(_QWORD *)(a1 + 32);
          v18 = v10[-1].m128i_i64[0];
        }
      }
      goto LABEL_12;
    }
    if ( v14 != v16 )
    {
      if ( v14 >= v16 )
        goto LABEL_9;
LABEL_38:
      v46 = *(_QWORD *)(a1 + 8);
      v47 = *(_QWORD *)a1;
      v48 = *(_DWORD *)(a1 + 16);
      *(__m128i *)a1 = _mm_loadu_si128(v12);
      *(_QWORD *)(a1 + 16) = v12[1].m128i_i64[0];
      v12->m128i_i64[0] = v47;
      v12->m128i_i64[1] = v46;
      v12[1].m128i_i32[0] = v48;
      v20 = *(_QWORD *)(a1 + 32);
      v18 = v10[-1].m128i_i64[0];
      goto LABEL_12;
    }
    if ( v12[1].m128i_i32[0] < (unsigned __int32)v10[-1].m128i_i32[2] )
      goto LABEL_38;
LABEL_9:
    if ( v13 == v16 )
    {
      if ( *(_DWORD *)(a1 + 40) < v10[-1].m128i_i32[2] )
        goto LABEL_11;
LABEL_40:
      v49 = _mm_loadu_si128((const __m128i *)(a1 + 24));
      v20 = *(_QWORD *)(a1 + 8);
      v50 = *(_DWORD *)(a1 + 16);
      v51 = *(_QWORD *)(a1 + 40);
      *(_QWORD *)(a1 + 24) = *(_QWORD *)a1;
      *(_QWORD *)(a1 + 32) = v20;
      *(_QWORD *)(a1 + 16) = v51;
      *(_DWORD *)(a1 + 40) = v50;
      *(__m128i *)a1 = v49;
      v18 = v10[-1].m128i_i64[0];
      goto LABEL_12;
    }
    if ( v13 >= v16 )
      goto LABEL_40;
LABEL_11:
    v17 = *(_QWORD *)a1;
    v18 = *(_QWORD *)(a1 + 8);
    v19 = *(_DWORD *)(a1 + 16);
    *(__m128i *)a1 = _mm_loadu_si128((__m128i *)((char *)v10 - 24));
    *(_QWORD *)(a1 + 16) = v10[-1].m128i_i64[1];
    v10[-2].m128i_i64[1] = v17;
    v10[-1].m128i_i64[0] = v18;
    v10[-1].m128i_i32[2] = v19;
    v20 = *(_QWORD *)(a1 + 32);
LABEL_12:
    v21 = v11;
    v22 = v10;
    v23 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 72LL);
    while ( 1 )
    {
      v24 = *(_DWORD *)(v20 + 72);
      v7 = (__m128i *)v21;
      v25 = v24 < v23;
      if ( v24 == v23 )
        v25 = *(_DWORD *)(v21 + 16) < *(_DWORD *)(a1 + 16);
      if ( !v25 )
        break;
LABEL_14:
      v20 = *(_QWORD *)(v21 + 32);
      v21 += 24LL;
    }
    for ( i = (__m128i *)((char *)v22 - 24); ; v18 = i->m128i_i64[1] )
    {
      v27 = *(_DWORD *)(v18 + 72);
      v22 = i;
      v28 = v27 > v23;
      if ( v27 == v23 )
        v28 = *(_DWORD *)(a1 + 16) < i[1].m128i_i32[0];
      i = (__m128i *)((char *)i - 24);
      if ( !v28 )
        break;
    }
    if ( v21 < (unsigned __int64)v22 )
    {
      v29 = *(_QWORD *)v21;
      v30 = *(_DWORD *)(v21 + 16);
      *(__m128i *)v21 = _mm_loadu_si128(v22);
      *(_QWORD *)(v21 + 16) = v22[1].m128i_i64[0];
      v22->m128i_i64[0] = v29;
      v18 = v22[-1].m128i_i64[0];
      v22->m128i_i64[1] = v20;
      v22[1].m128i_i32[0] = v30;
      v23 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 72LL);
      goto LABEL_14;
    }
    sub_371E870(v21, v10, v8);
    result = v21 - a1;
    if ( (__int64)(v21 - a1) > 384 )
    {
      if ( v8 )
      {
        v10 = (__m128i *)v21;
        continue;
      }
LABEL_34:
      v36 = (unsigned __int64)v7;
      v37 = v7;
      v38 = (__m128i *)((char *)v7 - 24);
      sub_371E770((const __m128i *)a1, v37, v36, a4, a5, a6);
      do
      {
        v42 = v38[1].m128i_i64[0];
        v43 = (__int64)v38->m128i_i64 - a1;
        v44 = _mm_loadu_si128(v38);
        *v38 = _mm_loadu_si128((const __m128i *)a1);
        v45 = (__int64)v38->m128i_i64 - a1;
        v38 = (__m128i *)((char *)v38 - 24);
        v38[2].m128i_i64[1] = *(_QWORD *)(a1 + 16);
        result = sub_371CEF0(
                   a1,
                   0,
                   0xAAAAAAAAAAAAAAABLL * (v45 >> 3),
                   v39,
                   v40,
                   v41,
                   v44.m128i_i64[0],
                   v44.m128i_i64[1],
                   v42);
      }
      while ( v43 > 24 );
    }
    return result;
  }
}
