// Function: sub_18B7900
// Address: 0x18b7900
//
__int64 __fastcall sub_18B7900(__int64 a1, const __m128i *a2)
{
  __m128i v4; // xmm0
  unsigned int v5; // esi
  __int64 v6; // r9
  __int64 v7; // r12
  int v8; // r11d
  unsigned __int32 i; // eax
  __int64 v10; // r10
  __int64 v11; // rdi
  unsigned __int32 v12; // eax
  __int64 v13; // rax
  int v15; // eax
  int v16; // edx
  __m128i v17; // xmm1
  __m128i *v18; // r14
  __m128i v19; // xmm0
  _QWORD *v20; // rdi
  __m128i *v21; // rdx
  _QWORD *v22; // rax
  __int32 v23; // r8d
  __int64 v24; // r8
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rdx
  bool v27; // cf
  unsigned __int64 v28; // rax
  __int64 v29; // r15
  __int64 v30; // rax
  __int64 m128i_i64; // rdx
  __m128i *v32; // rax
  __int64 v33; // rdi
  __m128i v34; // xmm3
  __m128i *v35; // r8
  __int64 v36; // rdi
  __int64 v37; // rdi
  __int64 v38; // rdi
  __int64 v39; // rdi
  _QWORD *v40; // rdi
  __int32 v41; // r9d
  const __m128i *v42; // r13
  __m128i *j; // r15
  __m128i *v44; // rdi
  __int8 v45; // si
  __int64 v46; // rsi
  __int32 v47; // r9d
  _QWORD *v48; // rdi
  __int64 v49; // rdi
  __int64 v50; // rdi
  const __m128i *v51; // rdi
  __int64 v52; // [rsp+0h] [rbp-160h]
  __int64 v53; // [rsp+8h] [rbp-158h]
  const __m128i *v54; // [rsp+10h] [rbp-150h]
  __int64 v55; // [rsp+10h] [rbp-150h]
  __m128i *v56; // [rsp+18h] [rbp-148h]
  __m128i v57; // [rsp+20h] [rbp-140h] BYREF
  int v58; // [rsp+30h] [rbp-130h]
  __int64 v59; // [rsp+40h] [rbp-120h]
  __int64 v60; // [rsp+48h] [rbp-118h]
  __int64 v61; // [rsp+50h] [rbp-110h]
  __int64 v62; // [rsp+58h] [rbp-108h]
  __int64 v63; // [rsp+60h] [rbp-100h]
  __int64 v64; // [rsp+68h] [rbp-F8h]
  __int64 v65; // [rsp+70h] [rbp-F0h]
  _QWORD v66[6]; // [rsp+80h] [rbp-E0h] BYREF
  __m128i v67; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v68; // [rsp+C0h] [rbp-A0h]
  __int64 v69; // [rsp+C8h] [rbp-98h]
  __int64 v70; // [rsp+D0h] [rbp-90h]
  __int64 v71; // [rsp+D8h] [rbp-88h]
  __int64 v72; // [rsp+E0h] [rbp-80h]
  __int64 v73; // [rsp+E8h] [rbp-78h]
  __int64 v74; // [rsp+F0h] [rbp-70h]
  __int64 v75; // [rsp+100h] [rbp-60h] BYREF
  _QWORD *v76; // [rsp+108h] [rbp-58h]
  __int64 *v77; // [rsp+110h] [rbp-50h]
  __int64 *v78; // [rsp+118h] [rbp-48h]
  __int64 v79; // [rsp+120h] [rbp-40h]

  v4 = _mm_loadu_si128(a2);
  v5 = *(_DWORD *)(a1 + 24);
  v58 = 0;
  v67 = v4;
  v57 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
LABEL_29:
    v5 *= 2;
    goto LABEL_30;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 0;
  v8 = 1;
  for ( i = (v5 - 1)
          & ((37 * v57.m128i_i32[2])
           ^ ((unsigned __int32)v57.m128i_i32[0] >> 9)
           ^ ((unsigned __int32)v57.m128i_i32[0] >> 4)); ; i = (v5 - 1) & v12 )
  {
    v10 = v6 + 24LL * i;
    v11 = *(_QWORD *)v10;
    if ( *(_OWORD *)&v57 == *(_OWORD *)v10 )
    {
      v13 = *(unsigned int *)(v10 + 16);
      return *(_QWORD *)(a1 + 32) + 120 * v13 + 16;
    }
    if ( v11 == -4 )
      break;
    if ( v11 == -8 && *(_QWORD *)(v10 + 8) == -2 && !v7 )
      v7 = v6 + 24LL * i;
LABEL_6:
    v12 = v8 + i;
    ++v8;
  }
  if ( *(_QWORD *)(v10 + 8) != -1 )
    goto LABEL_6;
  v15 = *(_DWORD *)(a1 + 16);
  if ( !v7 )
    v7 = v10;
  ++*(_QWORD *)a1;
  v16 = v15 + 1;
  if ( 4 * (v15 + 1) >= 3 * v5 )
    goto LABEL_29;
  if ( v5 - *(_DWORD *)(a1 + 20) - v16 <= v5 >> 3 )
  {
LABEL_30:
    sub_18B5B40(a1, v5);
    sub_18B4850(a1, v57.m128i_i64, (__int64 **)&v67);
    v7 = v67.m128i_i64[0];
    v16 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v16;
  if ( *(_QWORD *)v7 != -4 || *(_QWORD *)(v7 + 8) != -1 )
    --*(_DWORD *)(a1 + 20);
  v17 = _mm_loadu_si128(&v57);
  v59 = 0;
  v60 = 0;
  *(__m128i *)v7 = v17;
  v61 = 0;
  *(_DWORD *)(v7 + 16) = v58;
  v18 = *(__m128i **)(a1 + 40);
  v19 = _mm_loadu_si128(a2);
  v62 = 1;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66[0] = 0;
  v66[1] = 0;
  v66[2] = v66;
  v66[3] = v66;
  v66[4] = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 1;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v77 = &v75;
  v78 = &v75;
  v79 = 0;
  v67 = v19;
  if ( v18 != *(__m128i **)(a1 + 48) )
  {
    v20 = 0;
    if ( v18 )
    {
      *v18 = v19;
      v21 = v18 + 5;
      v18[1].m128i_i64[0] = v68;
      v18[1].m128i_i64[1] = v69;
      v18[2].m128i_i64[0] = v70;
      v70 = 0;
      v69 = 0;
      v68 = 0;
      v18[2].m128i_i16[4] = v71;
      v18[3].m128i_i64[0] = v72;
      v18[3].m128i_i64[1] = v73;
      v18[4].m128i_i64[0] = v74;
      v22 = v76;
      v74 = 0;
      v73 = 0;
      v72 = 0;
      if ( v76 )
      {
        v23 = v75;
        v18[5].m128i_i64[1] = (__int64)v76;
        v18[5].m128i_i32[0] = v23;
        v18[6].m128i_i64[0] = (__int64)v77;
        v18[6].m128i_i64[1] = (__int64)v78;
        v22[1] = v21;
        v18[7].m128i_i64[0] = v79;
        v18 = *(__m128i **)(a1 + 40);
        v76 = 0;
        v77 = &v75;
        v78 = &v75;
        v79 = 0;
      }
      else
      {
        v18[5].m128i_i32[0] = 0;
        v18[5].m128i_i64[1] = 0;
        v18[6].m128i_i64[0] = (__int64)v21;
        v18[6].m128i_i64[1] = (__int64)v21;
        v18[7].m128i_i64[0] = 0;
        v18 = *(__m128i **)(a1 + 40);
        v20 = v76;
      }
    }
    *(_QWORD *)(a1 + 40) = (char *)v18 + 120;
    goto LABEL_21;
  }
  v24 = (__int64)v18->m128i_i64 - *(_QWORD *)(a1 + 32);
  v54 = *(const __m128i **)(a1 + 32);
  v25 = 0xEEEEEEEEEEEEEEEFLL * (v24 >> 3);
  if ( v25 == 0x111111111111111LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v26 = 1;
  if ( v25 )
    v26 = 0xEEEEEEEEEEEEEEEFLL * (v24 >> 3);
  v27 = __CFADD__(v26, v25);
  v28 = v26 - 0x1111111111111111LL * (v24 >> 3);
  if ( v27 )
  {
    v29 = 0x7FFFFFFFFFFFFFF8LL;
    goto LABEL_39;
  }
  if ( v28 )
  {
    if ( v28 > 0x111111111111111LL )
      v28 = 0x111111111111111LL;
    v29 = 120 * v28;
LABEL_39:
    v52 = (__int64)v18->m128i_i64 - *(_QWORD *)(a1 + 32);
    v30 = sub_22077B0(v29);
    v24 = v52;
    v56 = (__m128i *)v30;
    m128i_i64 = v30 + 120;
    v53 = v30 + v29;
  }
  else
  {
    v53 = 0;
    m128i_i64 = 120;
    v56 = 0;
  }
  v32 = (__m128i *)((char *)v56 + v24);
  if ( &v56->m128i_i8[v24] )
  {
    v33 = v68;
    v34 = _mm_loadu_si128(&v67);
    v35 = v32 + 5;
    v68 = 0;
    v32[1].m128i_i64[0] = v33;
    v36 = v69;
    *v32 = v34;
    v32[1].m128i_i64[1] = v36;
    v69 = 0;
    v32[2].m128i_i64[0] = v70;
    v70 = 0;
    v32[2].m128i_i16[4] = v71;
    v37 = v72;
    v72 = 0;
    v32[3].m128i_i64[0] = v37;
    v38 = v73;
    v73 = 0;
    v32[3].m128i_i64[1] = v38;
    v39 = v74;
    v74 = 0;
    v32[4].m128i_i64[0] = v39;
    v40 = v76;
    if ( v76 )
    {
      v41 = v75;
      v32[5].m128i_i64[1] = (__int64)v76;
      v32[5].m128i_i32[0] = v41;
      v32[6].m128i_i64[0] = (__int64)v77;
      v32[6].m128i_i64[1] = (__int64)v78;
      v40[1] = v35;
      v76 = 0;
      v32[7].m128i_i64[0] = v79;
      v77 = &v75;
      v78 = &v75;
      v79 = 0;
    }
    else
    {
      v32[5].m128i_i32[0] = 0;
      v32[5].m128i_i64[1] = 0;
      v32[6].m128i_i64[0] = (__int64)v35;
      v32[6].m128i_i64[1] = (__int64)v35;
      v32[7].m128i_i64[0] = 0;
    }
  }
  v42 = v54;
  if ( v18 != v54 )
  {
    for ( j = v56; ; j = (__m128i *)((char *)j + 120) )
    {
      if ( j )
      {
        v44 = j + 5;
        *j = _mm_loadu_si128(v42);
        j[1].m128i_i64[0] = v42[1].m128i_i64[0];
        j[1].m128i_i64[1] = v42[1].m128i_i64[1];
        j[2].m128i_i64[0] = v42[2].m128i_i64[0];
        v45 = v42[2].m128i_i8[8];
        v42[2].m128i_i64[0] = 0;
        v42[1].m128i_i64[1] = 0;
        v42[1].m128i_i64[0] = 0;
        j[2].m128i_i8[8] = v45;
        j[2].m128i_i8[9] = v42[2].m128i_i8[9];
        j[3].m128i_i64[0] = v42[3].m128i_i64[0];
        j[3].m128i_i64[1] = v42[3].m128i_i64[1];
        j[4].m128i_i64[0] = v42[4].m128i_i64[0];
        v46 = v42[5].m128i_i64[1];
        v42[4].m128i_i64[0] = 0;
        v42[3].m128i_i64[1] = 0;
        v42[3].m128i_i64[0] = 0;
        if ( v46 )
        {
          v47 = v42[5].m128i_i32[0];
          j[5].m128i_i64[1] = v46;
          j[5].m128i_i32[0] = v47;
          j[6].m128i_i64[0] = v42[6].m128i_i64[0];
          j[6].m128i_i64[1] = v42[6].m128i_i64[1];
          *(_QWORD *)(v46 + 8) = v44;
          v48 = 0;
          j[7].m128i_i64[0] = v42[7].m128i_i64[0];
          v42[5].m128i_i64[1] = 0;
          v42[6].m128i_i64[0] = (__int64)v42[5].m128i_i64;
          v42[6].m128i_i64[1] = (__int64)v42[5].m128i_i64;
          v42[7].m128i_i64[0] = 0;
          goto LABEL_47;
        }
        j[5].m128i_i32[0] = 0;
        j[5].m128i_i64[1] = 0;
        j[6].m128i_i64[0] = (__int64)v44;
        j[6].m128i_i64[1] = (__int64)v44;
        j[7].m128i_i64[0] = 0;
      }
      v48 = (_QWORD *)v42[5].m128i_i64[1];
LABEL_47:
      sub_18B4EC0(v48);
      v49 = v42[3].m128i_i64[0];
      if ( v49 )
        j_j___libc_free_0(v49, v42[4].m128i_i64[0] - v49);
      v50 = v42[1].m128i_i64[0];
      if ( v50 )
        j_j___libc_free_0(v50, v42[2].m128i_i64[0] - v50);
      v42 = (const __m128i *)((char *)v42 + 120);
      if ( v18 == v42 )
      {
        m128i_i64 = (__int64)j[15].m128i_i64;
        break;
      }
    }
  }
  v51 = v54;
  if ( v54 )
  {
    v55 = m128i_i64;
    j_j___libc_free_0(v51, *(_QWORD *)(a1 + 48) - (_QWORD)v51);
    m128i_i64 = v55;
  }
  v20 = v76;
  *(_QWORD *)(a1 + 40) = m128i_i64;
  *(_QWORD *)(a1 + 32) = v56;
  *(_QWORD *)(a1 + 48) = v53;
LABEL_21:
  sub_18B4EC0(v20);
  if ( v72 )
    j_j___libc_free_0(v72, v74 - v72);
  if ( v68 )
    j_j___libc_free_0(v68, v70 - v68);
  sub_18B4EC0(0);
  v13 = -286331153 * (unsigned int)((__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 3) - 1;
  *(_DWORD *)(v7 + 16) = v13;
  return *(_QWORD *)(a1 + 32) + 120 * v13 + 16;
}
