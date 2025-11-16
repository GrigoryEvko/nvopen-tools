// Function: sub_8453D0
// Address: 0x8453d0
//
__int64 __fastcall sub_8453D0(
        __m128i *a1,
        __m128i *a2,
        _DWORD *a3,
        _BYTE *a4,
        unsigned int a5,
        unsigned int a6,
        unsigned int a7,
        unsigned int a8,
        FILE *a9)
{
  _BYTE *v9; // r12
  __int64 result; // rax
  __int64 v11; // r15
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  __m128i *v16; // rcx
  __m128i *v17; // rbx
  __int8 v18; // al
  __m128i *v19; // rcx
  __int64 v20; // rdx
  char v21; // al
  char v22; // bl
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  _QWORD *v31; // rbx
  unsigned __int8 v32; // bl
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __m128i *v37; // rax
  __m128i v38; // xmm1
  __m128i v39; // xmm2
  __m128i v40; // xmm3
  __m128i v41; // xmm4
  __m128i v42; // xmm5
  __m128i v43; // xmm6
  __m128i v44; // xmm7
  __m128i v45; // xmm0
  __int8 v46; // al
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __m128i v51; // xmm2
  __m128i v52; // xmm3
  __m128i v53; // xmm4
  __m128i v54; // xmm5
  __m128i v55; // xmm6
  __m128i v56; // xmm7
  __m128i v57; // xmm1
  __m128i v58; // xmm2
  __m128i v59; // xmm3
  __m128i v60; // xmm4
  __m128i v61; // xmm5
  __m128i v62; // xmm6
  int v63; // [rsp+8h] [rbp-1D8h]
  unsigned int v64; // [rsp+14h] [rbp-1CCh] BYREF
  __m128i *v65; // [rsp+18h] [rbp-1C8h] BYREF
  _BYTE v66[48]; // [rsp+20h] [rbp-1C0h] BYREF
  __m128i v67[9]; // [rsp+50h] [rbp-190h] BYREF
  __m128i v68; // [rsp+E0h] [rbp-100h]
  __m128i v69; // [rsp+F0h] [rbp-F0h]
  __m128i v70; // [rsp+100h] [rbp-E0h]
  __m128i v71; // [rsp+110h] [rbp-D0h]
  __m128i v72; // [rsp+120h] [rbp-C0h]
  __m128i v73; // [rsp+130h] [rbp-B0h]
  __m128i v74; // [rsp+140h] [rbp-A0h]
  __m128i v75; // [rsp+150h] [rbp-90h]
  __m128i v76; // [rsp+160h] [rbp-80h]
  __m128i v77; // [rsp+170h] [rbp-70h]
  __m128i v78; // [rsp+180h] [rbp-60h]
  __m128i v79; // [rsp+190h] [rbp-50h]
  __m128i v80; // [rsp+1A0h] [rbp-40h]

  if ( !a4 || (v9 = a4, (a4[16] & 0x88) != 0) )
  {
    v9 = v66;
    result = sub_840D60(a1, a2, a3, (__int64)a2, a5, a6, 0, a7, a8, a9, (__int64)v66, 0);
    if ( !(_DWORD)result )
      return result;
  }
  else if ( !*(_QWORD *)a4 )
  {
    sub_6F69D0(a1, 8u);
  }
  if ( (a7 & 0x1000000) != 0 )
    v9[17] |= 2u;
  if ( (a7 & 0x200000) != 0 )
    v9[17] |= 4u;
  if ( (a7 & 4) == 0 )
  {
    v9[16] &= ~4u;
    return sub_845370(a1, a2, (__int64)v9);
  }
  v11 = a1->m128i_i64[0];
  v65 = (__m128i *)sub_724DC0();
  if ( dword_4F04C44 == -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0 )
    sub_6E6B60(a1, 1, (__int64)qword_4F04C68, v12, v13, v14);
  v15 = *(_QWORD *)v9;
  if ( *(_QWORD *)v9 && (*(_BYTE *)(v15 + 193) & 2) != 0 && *(_BYTE *)(v15 + 174) != 1 )
  {
    v37 = sub_73D790(*(_QWORD *)(v15 + 152));
    v38 = _mm_loadu_si128(a1 + 1);
    v39 = _mm_loadu_si128(a1 + 2);
    v40 = _mm_loadu_si128(a1 + 3);
    v17 = v37;
    v41 = _mm_loadu_si128(a1 + 4);
    v42 = _mm_loadu_si128(a1 + 5);
    v67[0] = _mm_loadu_si128(a1);
    v43 = _mm_loadu_si128(a1 + 6);
    v44 = _mm_loadu_si128(a1 + 7);
    v67[1] = v38;
    v45 = _mm_loadu_si128(a1 + 8);
    v46 = a1[1].m128i_i8[0];
    v67[2] = v39;
    v67[3] = v40;
    v67[4] = v41;
    v67[5] = v42;
    v67[6] = v43;
    v67[7] = v44;
    v67[8] = v45;
    if ( v46 == 2 )
    {
      v51 = _mm_loadu_si128(a1 + 10);
      v52 = _mm_loadu_si128(a1 + 11);
      v53 = _mm_loadu_si128(a1 + 12);
      v54 = _mm_loadu_si128(a1 + 13);
      v68 = _mm_loadu_si128(a1 + 9);
      v55 = _mm_loadu_si128(a1 + 14);
      v56 = _mm_loadu_si128(a1 + 15);
      v69 = v51;
      v57 = _mm_loadu_si128(a1 + 16);
      v58 = _mm_loadu_si128(a1 + 17);
      v70 = v52;
      v59 = _mm_loadu_si128(a1 + 18);
      v71 = v53;
      v60 = _mm_loadu_si128(a1 + 19);
      v72 = v54;
      v61 = _mm_loadu_si128(a1 + 20);
      v73 = v55;
      v62 = _mm_loadu_si128(a1 + 21);
      v74 = v56;
      v75 = v57;
      v76 = v58;
      v77 = v59;
      v78 = v60;
      v79 = v61;
      v80 = v62;
    }
    else if ( v46 == 5 || v46 == 1 )
    {
      v68.m128i_i64[0] = a1[9].m128i_i64[0];
    }
    sub_8449E0(v67, v17, (__int64)v9, 0, 0);
    sub_6E6B60(v67, 1, v47, v48, v49, v50);
    v16 = v67;
  }
  else
  {
    v16 = a1;
    v17 = (__m128i *)v11;
  }
  v18 = v16[1].m128i_i8[0];
  if ( v18 == 2 )
  {
    v19 = v16 + 9;
    v20 = 1;
  }
  else if ( v18 == 1 && (unsigned int)sub_719770(v16[9].m128i_i64[0], (__int64)v65, 1u, 1u) )
  {
    v19 = v65;
    v20 = 1;
  }
  else
  {
    v19 = 0;
    v20 = 0;
  }
  v21 = v9[17];
  if ( (v21 & 2) != 0 )
  {
LABEL_22:
    v22 = v21 & 1;
    sub_724E30((__int64)&v65);
    v9[16] &= ~4u;
    sub_845370(a1, a2, (__int64)v9);
    result = sub_6E6B60(a1, 1, v23, v24, v25, v26);
    if ( a1[1].m128i_i8[0] != 2 )
    {
      if ( !v22 )
        return result;
      result = sub_6F4B70(a1, 1, v27, v28, v29, v30);
      if ( a1[1].m128i_i8[0] != 2 )
        return result;
    }
    goto LABEL_25;
  }
  v63 = v20;
  if ( (unsigned int)sub_8DD690(v9 + 24, v17, v20, v19, a2, &v64) || !v63 && (unsigned int)sub_696840((__int64)a1) )
  {
    v21 = v9[17];
    goto LABEL_22;
  }
  if ( sub_6E53E0(7, v64, a9) )
  {
    v32 = 7;
    if ( dword_4F077BC && !(_DWORD)qword_4F077B4 )
      v32 = qword_4F077A8 < 0x15F90u ? 5 : 7;
    sub_686040(v32, v64, a9, v11, (__int64)a2);
    if ( sub_67D370((int *)v64, v32, a9) )
      sub_6E6260(a1);
  }
  sub_724E30((__int64)&v65);
  v9[16] &= ~4u;
  sub_845370(a1, a2, (__int64)v9);
  result = sub_6E6B60(a1, 1, v33, v34, v35, v36);
  if ( a1[1].m128i_i8[0] == 2 )
  {
LABEL_25:
    if ( a1[19].m128i_i8[13] == 12 && a1[20].m128i_i8[0] == 1 )
    {
      result = (__int64)sub_72E9A0((__int64)a1[9].m128i_i64);
      v31 = (_QWORD *)result;
      if ( *(_BYTE *)(result + 24) == 1 && (*(_BYTE *)(result + 27) & 2) != 0 )
      {
        result = sub_730740(result);
        if ( (_DWORD)result )
        {
          result = sub_8DBE70(*v31);
          if ( !(_DWORD)result )
            a1[20].m128i_i8[1] |= 0x20u;
        }
      }
    }
  }
  return result;
}
