// Function: sub_20461E0
// Address: 0x20461e0
//
__int64 __fastcall sub_20461E0(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  unsigned __int64 v7; // r15
  __int64 v9; // rbx
  __m128i *v10; // r9
  __int64 v11; // r12
  unsigned __int32 v12; // ecx
  __m128i *v13; // rax
  unsigned __int32 v14; // edx
  bool v15; // cf
  bool v16; // di
  unsigned __int32 v17; // esi
  bool v18; // cf
  bool v19; // zf
  bool v20; // r13
  __int64 v21; // r11
  __int64 v22; // rdi
  __int32 v23; // edx
  __int32 v24; // r10d
  bool v25; // cf
  bool v26; // zf
  bool v27; // al
  unsigned __int32 v28; // edx
  unsigned __int32 v29; // edi
  unsigned __int32 v30; // esi
  unsigned __int64 v31; // r13
  __m128i *v32; // rcx
  bool v33; // cf
  bool i; // al
  __m128i *j; // rax
  bool v36; // cf
  bool v37; // zf
  unsigned __int32 v38; // r8d
  bool v39; // dl
  __int64 v40; // rsi
  __int64 v41; // rdx
  __int32 v42; // eax
  unsigned int v43; // r11d
  bool v44; // cf
  bool v45; // zf
  bool v46; // r13
  __int64 v47; // r11
  __int64 v48; // r10
  __int32 v49; // edi
  __int32 v50; // ecx
  bool v51; // cf
  bool v52; // zf
  bool v53; // dl
  unsigned __int32 v54; // r11d
  __int64 v55; // rbx
  __int64 k; // r12
  __m128i *v57; // r15
  __int64 v58; // rax
  __int64 v59; // r12
  __m128i v60; // xmm1
  __int64 v61; // rdx
  unsigned __int32 v62; // r11d
  unsigned __int32 v63; // r11d
  __m128i v64; // xmm7
  __int64 v65; // rax
  __m128i v66; // xmm6
  __int64 v67; // rax
  unsigned __int32 v68; // r15d
  unsigned __int32 v69; // esi
  __m128i v70; // [rsp-58h] [rbp-58h]

  result = (__int64)a2->m128i_i64 - a1;
  if ( (__int64)a2->m128i_i64 - a1 <= 384 )
    return result;
  v7 = (unsigned __int64)a2;
  v9 = a3;
  if ( !a3 )
    goto LABEL_41;
  v10 = a2;
  v11 = a1 + 24;
  while ( 2 )
  {
    v12 = *(_DWORD *)(a1 + 44);
    --v9;
    v13 = (__m128i *)(a1
                    + 8
                    * ((__int64)(0xAAAAAAAAAAAAAAABLL * (((__int64)v10->m128i_i64 - a1) >> 3)) / 2
                     + ((0xAAAAAAAAAAAAAAABLL * (((__int64)v10->m128i_i64 - a1) >> 3)
                       + ((0xAAAAAAAAAAAAAAABLL * (((__int64)v10->m128i_i64 - a1) >> 3)) >> 63))
                      & 0xFFFFFFFFFFFFFFFELL)));
    v14 = v13[1].m128i_u32[1];
    v15 = v14 < v12;
    if ( v14 != v12 )
      goto LABEL_5;
    v54 = v13[1].m128i_u32[0];
    if ( *(_DWORD *)(a1 + 40) == v54 )
    {
      v15 = *(_QWORD *)(a1 + 24) < v13->m128i_i64[0];
LABEL_5:
      v16 = v15;
      goto LABEL_6;
    }
    v16 = *(_DWORD *)(a1 + 40) > v54;
LABEL_6:
    v17 = v10[-1].m128i_u32[3];
    if ( v16 )
    {
      v18 = v14 < v17;
      v19 = v14 == v17;
      if ( v14 == v17 && (v63 = v10[-1].m128i_u32[2], v18 = v13[1].m128i_i32[0] < v63, v19 = v13[1].m128i_i32[0] == v63) )
        v20 = v13->m128i_i64[0] < (unsigned __int64)v10[-2].m128i_i64[1];
      else
        v20 = !v18 && !v19;
      v21 = *(_QWORD *)a1;
      v22 = *(_QWORD *)(a1 + 8);
      v23 = *(_DWORD *)(a1 + 16);
      v24 = *(_DWORD *)(a1 + 20);
      if ( v20 )
      {
        *(__m128i *)a1 = _mm_loadu_si128(v13);
        *(_QWORD *)(a1 + 16) = v13[1].m128i_i64[0];
        v13->m128i_i64[0] = v21;
        v13->m128i_i64[1] = v22;
        v13[1].m128i_i32[0] = v23;
        v13[1].m128i_i32[1] = v24;
        v29 = *(_DWORD *)(a1 + 44);
        v28 = v10[-1].m128i_u32[3];
      }
      else
      {
        v25 = v12 < v17;
        v26 = v12 == v17;
        if ( v12 == v17
          && (v69 = v10[-1].m128i_u32[2], v25 = *(_DWORD *)(a1 + 40) < v69, v26 = *(_DWORD *)(a1 + 40) == v69) )
        {
          v27 = *(_QWORD *)(a1 + 24) < v10[-2].m128i_i64[1];
        }
        else
        {
          v27 = !v25 && !v26;
        }
        if ( v27 )
        {
          *(__m128i *)a1 = _mm_loadu_si128((__m128i *)((char *)v10 - 24));
          *(_QWORD *)(a1 + 16) = v10[-1].m128i_i64[1];
          v10[-1].m128i_i32[2] = v23;
          v28 = v24;
          v10[-2].m128i_i64[1] = v21;
          v10[-1].m128i_i64[0] = v22;
          v10[-1].m128i_i32[3] = v24;
          v29 = *(_DWORD *)(a1 + 44);
        }
        else
        {
          v66 = _mm_loadu_si128((const __m128i *)(a1 + 24));
          v67 = *(_QWORD *)(a1 + 40);
          *(_QWORD *)(a1 + 32) = v22;
          v29 = v24;
          *(_QWORD *)(a1 + 24) = v21;
          *(_QWORD *)(a1 + 16) = v67;
          *(_DWORD *)(a1 + 40) = v23;
          *(_DWORD *)(a1 + 44) = v24;
          *(__m128i *)a1 = v66;
          v28 = v10[-1].m128i_u32[3];
        }
      }
    }
    else
    {
      v44 = v12 < v17;
      v45 = v12 == v17;
      if ( v12 == v17
        && (v62 = v10[-1].m128i_u32[2], v44 = *(_DWORD *)(a1 + 40) < v62, v45 = *(_DWORD *)(a1 + 40) == v62) )
      {
        v46 = *(_QWORD *)(a1 + 24) < v10[-2].m128i_i64[1];
      }
      else
      {
        v46 = !v44 && !v45;
      }
      v47 = *(_QWORD *)a1;
      v48 = *(_QWORD *)(a1 + 8);
      v49 = *(_DWORD *)(a1 + 16);
      v50 = *(_DWORD *)(a1 + 20);
      if ( v46 )
      {
        v64 = _mm_loadu_si128((const __m128i *)(a1 + 24));
        v65 = *(_QWORD *)(a1 + 40);
        *(_QWORD *)(a1 + 24) = v47;
        *(_DWORD *)(a1 + 40) = v49;
        v29 = v50;
        *(_QWORD *)(a1 + 16) = v65;
        *(_QWORD *)(a1 + 32) = v48;
        *(_DWORD *)(a1 + 44) = v50;
        *(__m128i *)a1 = v64;
        v28 = v10[-1].m128i_u32[3];
      }
      else
      {
        v51 = v14 < v17;
        v52 = v14 == v17;
        if ( v14 == v17
          && (v68 = v10[-1].m128i_u32[2], v51 = v13[1].m128i_i32[0] < v68, v52 = v13[1].m128i_i32[0] == v68) )
        {
          v53 = v13->m128i_i64[0] < (unsigned __int64)v10[-2].m128i_i64[1];
        }
        else
        {
          v53 = !v51 && !v52;
        }
        if ( v53 )
        {
          v28 = *(_DWORD *)(a1 + 20);
          *(__m128i *)a1 = _mm_loadu_si128((__m128i *)((char *)v10 - 24));
          *(_QWORD *)(a1 + 16) = v10[-1].m128i_i64[1];
          v10[-2].m128i_i64[1] = v47;
          v10[-1].m128i_i64[0] = v48;
          v10[-1].m128i_i32[2] = v49;
          v10[-1].m128i_i32[3] = v50;
          v29 = *(_DWORD *)(a1 + 44);
        }
        else
        {
          *(__m128i *)a1 = _mm_loadu_si128(v13);
          *(_QWORD *)(a1 + 16) = v13[1].m128i_i64[0];
          v13->m128i_i64[0] = v47;
          v13->m128i_i64[1] = v48;
          v13[1].m128i_i32[0] = v49;
          v13[1].m128i_i32[1] = v50;
          v29 = *(_DWORD *)(a1 + 44);
          v28 = v10[-1].m128i_u32[3];
        }
      }
    }
    v30 = *(_DWORD *)(a1 + 20);
    v31 = v11;
    v32 = v10;
    v7 = v11;
    v33 = v30 < v29;
    if ( v30 == v29 )
      goto LABEL_26;
LABEL_15:
    for ( i = v33; i; i = *(_DWORD *)(v31 + 16) > v43 )
    {
LABEL_25:
      v29 = *(_DWORD *)(v31 + 44);
      v31 += 24LL;
      v7 = v31;
      v33 = v30 < v29;
      if ( v30 != v29 )
        goto LABEL_15;
LABEL_26:
      v43 = *(_DWORD *)(a1 + 16);
      if ( *(_DWORD *)(v31 + 16) == v43 )
      {
        v33 = *(_QWORD *)v31 < *(_QWORD *)a1;
        goto LABEL_15;
      }
    }
    for ( j = (__m128i *)((char *)v32 - 24); ; v28 = j[1].m128i_u32[1] )
    {
      v32 = j;
      v36 = v30 < v28;
      v37 = v30 == v28;
      if ( v30 == v28 )
      {
        v38 = j[1].m128i_u32[0];
        v36 = *(_DWORD *)(a1 + 16) < v38;
        v37 = *(_DWORD *)(a1 + 16) == v38;
        if ( *(_DWORD *)(a1 + 16) == v38 )
          break;
      }
      j = (__m128i *)((char *)j - 24);
      if ( v36 || v37 )
        goto LABEL_23;
LABEL_19:
      ;
    }
    v39 = *(_QWORD *)a1 < j->m128i_i64[0];
    j = (__m128i *)((char *)j - 24);
    if ( v39 )
      goto LABEL_19;
LABEL_23:
    if ( v31 < (unsigned __int64)v32 )
    {
      v40 = *(_QWORD *)v31;
      v41 = *(_QWORD *)(v31 + 8);
      v42 = *(_DWORD *)(v31 + 16);
      *(__m128i *)v31 = _mm_loadu_si128(v32);
      *(_QWORD *)(v31 + 16) = v32[1].m128i_i64[0];
      v32->m128i_i64[1] = v41;
      v28 = v32[-1].m128i_u32[3];
      v32->m128i_i64[0] = v40;
      v32[1].m128i_i32[0] = v42;
      v32[1].m128i_i32[1] = v29;
      v30 = *(_DWORD *)(a1 + 20);
      goto LABEL_25;
    }
    sub_20461E0(v31, v10, v9);
    result = v31 - a1;
    if ( (__int64)(v31 - a1) > 384 )
    {
      if ( v9 )
      {
        v10 = (__m128i *)v31;
        continue;
      }
LABEL_41:
      v55 = 0xAAAAAAAAAAAAAAABLL * (result >> 3);
      for ( k = (v55 - 2) >> 1; ; --k )
      {
        v70 = _mm_loadu_si128((const __m128i *)(a1 + 24 * k));
        sub_2045D00(a1, k, v55, a4, a5, a6, v70.m128i_u64[0], v70.m128i_i64[1], *(_QWORD *)(a1 + 24 * k + 16));
        if ( !k )
          break;
      }
      v57 = (__m128i *)(v7 - 24);
      do
      {
        v58 = v57[1].m128i_i64[0];
        v59 = (__int64)v57->m128i_i64 - a1;
        v60 = _mm_loadu_si128(v57);
        *v57 = _mm_loadu_si128((const __m128i *)a1);
        v61 = (__int64)v57->m128i_i64 - a1;
        v57 = (__m128i *)((char *)v57 - 24);
        v57[2].m128i_i64[1] = *(_QWORD *)(a1 + 16);
        result = sub_2045D00(
                   a1,
                   0,
                   0xAAAAAAAAAAAAAAABLL * (v61 >> 3),
                   a4,
                   a5,
                   a6,
                   v60.m128i_u64[0],
                   v60.m128i_i64[1],
                   v58);
      }
      while ( v59 > 24 );
    }
    return result;
  }
}
