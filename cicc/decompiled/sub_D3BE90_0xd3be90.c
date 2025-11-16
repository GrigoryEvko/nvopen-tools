// Function: sub_D3BE90
// Address: 0xd3be90
//
__m128i *__fastcall sub_D3BE90(__m128i *a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 v9; // r15
  __int64 v10; // r13
  unsigned __int64 v11; // r15
  unsigned __int8 *v12; // r14
  __int64 v13; // rax
  unsigned int v14; // edx
  __int64 v15; // rax
  unsigned int v16; // edx
  char v18; // dl
  __int64 v19; // rax
  char v20; // dl
  __int64 v21; // r15
  __int64 v22; // r11
  __int64 v23; // rsi
  char v24; // al
  __int64 v25; // r11
  char v26; // r10
  char v27; // al
  __int64 v28; // r15
  __int64 v29; // rdx
  __int64 v30; // rdx
  char v31; // al
  char v32; // al
  char v33; // al
  char v34; // al
  _QWORD *v35; // rbx
  __int64 v36; // rax
  __int64 v37; // rax
  char v38; // al
  __m128i v39; // rax
  __m128i v40; // rax
  char v41; // bl
  __int64 v42; // rax
  __int64 v43; // rdx
  char v44; // bl
  __m128i v45; // rax
  __int64 v46; // rsi
  __int64 v47; // rcx
  unsigned __int64 v48; // rdx
  unsigned __int64 v49; // rax
  bool v50; // cl
  __m128i v51; // xmm0
  __m128i v52; // xmm1
  __m128i v53; // xmm2
  __m128i v54; // xmm3
  __int64 v55; // rax
  __int64 v56; // rax
  unsigned __int8 *v57; // rax
  __int64 v58; // rsi
  __int64 v59; // rax
  char v60; // r10
  __int64 v61; // rax
  __int64 v62; // rsi
  __int64 v63; // [rsp+18h] [rbp-118h]
  __int64 v64; // [rsp+20h] [rbp-110h]
  __int64 v65; // [rsp+28h] [rbp-108h]
  char v66; // [rsp+30h] [rbp-100h]
  char v67; // [rsp+30h] [rbp-100h]
  __int64 v68; // [rsp+38h] [rbp-F8h]
  __int64 v69; // [rsp+38h] [rbp-F8h]
  __int8 v70; // [rsp+38h] [rbp-F8h]
  char v71; // [rsp+38h] [rbp-F8h]
  char v72; // [rsp+47h] [rbp-E9h]
  __int8 v73; // [rsp+47h] [rbp-E9h]
  __int64 v74; // [rsp+48h] [rbp-E8h]
  __int64 v75; // [rsp+50h] [rbp-E0h]
  __int64 v76; // [rsp+50h] [rbp-E0h]
  char v77; // [rsp+50h] [rbp-E0h]
  unsigned __int64 v78; // [rsp+58h] [rbp-D8h]
  __int64 v79; // [rsp+58h] [rbp-D8h]
  __int64 v80; // [rsp+58h] [rbp-D8h]
  __int64 v81; // [rsp+58h] [rbp-D8h]
  __int64 v82; // [rsp+58h] [rbp-D8h]
  unsigned __int64 v83; // [rsp+58h] [rbp-D8h]
  char v84; // [rsp+60h] [rbp-D0h]
  unsigned __int64 v85; // [rsp+60h] [rbp-D0h]
  __int64 v86; // [rsp+68h] [rbp-C8h]
  __int64 v88; // [rsp+70h] [rbp-C0h]
  __int64 v89; // [rsp+70h] [rbp-C0h]
  unsigned __int8 *v91; // [rsp+78h] [rbp-B8h]
  __m128i v92; // [rsp+A0h] [rbp-90h] BYREF
  __m128i v93; // [rsp+C0h] [rbp-70h] BYREF
  __m128i v94; // [rsp+D0h] [rbp-60h] BYREF
  __m128i v95; // [rsp+E0h] [rbp-50h] BYREF
  __int64 v96; // [rsp+F0h] [rbp-40h]

  v86 = sub_AA4E30(**(_QWORD **)(a2[1] + 32));
  v9 = *a3;
  v10 = *a3 >> 2;
  v11 = v9 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (((unsigned __int8)(*a5 >> 2) | (unsigned __int8)v10) & 1) == 0 )
    goto LABEL_23;
  if ( *(_BYTE *)a4 == 61 )
  {
    v12 = *(unsigned __int8 **)(a4 + 8);
    if ( *(_BYTE *)a6 != 61 )
      goto LABEL_4;
  }
  else
  {
    v12 = *(unsigned __int8 **)(*(_QWORD *)(a4 - 64) + 8LL);
    if ( *(_BYTE *)a6 != 61 )
    {
LABEL_4:
      v91 = *(unsigned __int8 **)(*(_QWORD *)(a6 - 64) + 8LL);
      goto LABEL_5;
    }
  }
  v91 = *(unsigned __int8 **)(a6 + 8);
LABEL_5:
  v13 = *(_QWORD *)(v11 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17 <= 1 )
    v13 = **(_QWORD **)(v13 + 16);
  v14 = *(_DWORD *)(v13 + 8);
  v15 = *(_QWORD *)((*a5 & 0xFFFFFFFFFFFFFFF8LL) + 8);
  v16 = v14 >> 8;
  if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 <= 1 )
    v15 = **(_QWORD **)(v15 + 16);
  if ( v16 != *(_DWORD *)(v15 + 8) >> 8 )
  {
LABEL_10:
    a1->m128i_i32[0] = 1;
    a1[3].m128i_i8[8] = 0;
    return a1;
  }
  v75 = *a5 >> 2;
  v78 = *a5 & 0xFFFFFFFFFFFFFFF8LL;
  v88 = *(_QWORD *)(*a2 + 112);
  v74 = sub_D34EB0(*a2, v12, v11, a2[1], a2[2], 1, 1);
  v84 = v18;
  v19 = sub_D34EB0(*a2, v91, v78, a2[1], a2[2], 1, 1);
  v72 = v20;
  v65 = v19;
  v21 = sub_DEEF40(*a2, v11);
  v22 = sub_DEEF40(*a2, v78);
  if ( v84 && v74 < 0 )
  {
    if ( v72 )
    {
      v58 = v65;
      v65 = v74;
      v59 = v21;
      v21 = v22;
      v22 = v59;
      v57 = v12;
      v12 = v91;
      v74 = v58;
    }
    else
    {
      v65 = v74;
      v56 = v21;
      v21 = v22;
      v22 = v56;
      v57 = v12;
      v72 = 1;
      v12 = v91;
      v84 = 0;
    }
    v91 = v57;
  }
  v79 = v22;
  v23 = v21;
  v64 = sub_DCC810(v88, v22, v21, 0, 0);
  v24 = sub_DADE90(v88, v21, a2[1]);
  v25 = v79;
  v26 = v75;
  if ( v24 || (v23 = v79, v27 = sub_DADE90(v88, v79, a2[1]), v25 = v79, v26 = v75, v27) )
  {
    v66 = v26;
    v68 = v25;
    v80 = sub_DEF9D0(*a2, v23);
    v28 = sub_D3B9E0(a2[1], v21, (__int64)v12, v80, *(_QWORD *)(*a2 + 112), (__int64)(a2 + 45));
    v63 = v29;
    v81 = sub_D3B9E0(a2[1], v68, (__int64)v91, v80, *(_QWORD *)(*a2 + 112), (__int64)(a2 + 45));
    v76 = v30;
    v31 = sub_D96A50(v28);
    v26 = v66;
    if ( !v31 )
    {
      v32 = sub_D96A50(v63);
      v26 = v66;
      if ( !v32 )
      {
        v33 = sub_D96A50(v81);
        v26 = v66;
        if ( !v33 )
        {
          v34 = sub_D96A50(v76);
          v26 = v66;
          if ( !v34 )
          {
            if ( !*((_BYTE *)a2 + 440) )
            {
              sub_DE4EA0(&v93, a2[1], v88);
              v60 = v66;
              if ( *((_BYTE *)a2 + 440) )
              {
                v62 = *((unsigned int *)a2 + 104);
                *((_BYTE *)a2 + 440) = 0;
                sub_C7D6A0(a2[50], 16 * v62, 8);
                v60 = v66;
              }
              v61 = v93.m128i_i64[1];
              *((_BYTE *)a2 + 440) = 1;
              a2[49] = 1;
              a2[50] = v61;
              v71 = v60;
              a2[51] = v94.m128i_i64[0];
              ++v93.m128i_i64[0];
              *((_DWORD *)a2 + 104) = v94.m128i_i32[2];
              v93.m128i_i64[1] = 0;
              *((_WORD *)a2 + 212) = v95.m128i_i16[0];
              v94.m128i_i64[0] = 0;
              a2[54] = v95.m128i_i64[1];
              v94.m128i_i32[2] = 0;
              sub_C7D6A0(0, 0, 8);
              v26 = v71;
            }
            v35 = a2 + 49;
            v67 = v26;
            v69 = sub_DE2740(v88, v63, v35);
            v36 = sub_DE2740(v88, v81, v35);
            if ( (unsigned __int8)sub_DC3A60(v88, 37, v69, v36)
              || (v82 = sub_DE2740(v88, v76, v35),
                  v37 = sub_DE2740(v88, v28, v35),
                  v38 = sub_DC3A60(v88, 37, v82, v37),
                  v26 = v67,
                  v38) )
            {
LABEL_23:
              a1->m128i_i32[0] = 0;
              a1[3].m128i_i8[8] = 0;
              return a1;
            }
          }
        }
      }
    }
  }
  if ( !v84 || !v72 )
  {
    a1->m128i_i32[0] = 2;
    a1[3].m128i_i8[8] = 0;
    return a1;
  }
  if ( !v74 || !v65 || v74 > 0 != v65 > 0 )
    goto LABEL_10;
  v77 = v26;
  v39.m128i_i64[0] = sub_9208B0(v86, (__int64)v12);
  v93 = v39;
  v73 = v39.m128i_i8[8];
  v85 = v39.m128i_i64[0] + 7;
  v40.m128i_i64[0] = sub_9208B0(v86, (__int64)v91);
  v83 = v40.m128i_i64[0] + 7;
  v70 = v40.m128i_i8[8];
  v93 = v40;
  v41 = sub_AE5020(v86, (__int64)v12);
  v42 = sub_9208B0(v86, (__int64)v12);
  v93.m128i_i64[1] = v43;
  v93.m128i_i64[0] = ((1LL << v41) + ((unsigned __int64)(v42 + 7) >> 3) - 1) >> v41 << v41;
  v89 = sub_CA1930(&v93);
  v44 = sub_AE5020(v86, (__int64)v91);
  v45.m128i_i64[0] = ((1LL << v44) + ((unsigned __int64)(sub_9208B0(v86, (__int64)v91) + 7) >> 3) - 1) >> v44 << v44;
  v93 = v45;
  v46 = 0;
  v47 = sub_CA1930(&v93);
  if ( v83 >> 3 == v85 >> 3 && v73 == v70 )
    v46 = v47;
  v92.m128i_i8[15] = 0;
  *(__int32 *)((char *)&v92.m128i_i32[2] + 1) = 0;
  v48 = v89 * abs64(v74);
  v49 = v47 * abs64(v65);
  *(__int16 *)((char *)&v92.m128i_i16[6] + 1) = 0;
  if ( v48 < v49 )
  {
    v48 = v49;
    v50 = 0;
    v49 = 0;
  }
  else
  {
    v50 = v48 == v49;
    if ( v48 != v49 )
      v49 = 0;
  }
  v92.m128i_i64[0] = v49;
  v92.m128i_i8[8] = v50;
  v51 = _mm_loadu_si128(&v92);
  v93.m128i_i64[0] = v64;
  v94.m128i_i64[1] = v51.m128i_i64[1];
  v95.m128i_i8[0] = v65 == v74;
  v94.m128i_i64[0] = v49;
  v93.m128i_i64[1] = v48;
  v52 = _mm_loadu_si128(&v93);
  v94.m128i_i8[8] = v50;
  v53 = _mm_loadu_si128(&v94);
  v95.m128i_i64[1] = v46;
  v54 = _mm_loadu_si128(&v95);
  LOBYTE(v96) = v10 & 1;
  BYTE1(v96) = v77 & 1;
  v55 = v96;
  a1[3].m128i_i8[8] = 1;
  a1[3].m128i_i64[0] = v55;
  *a1 = v52;
  a1[1] = v53;
  a1[2] = v54;
  return a1;
}
