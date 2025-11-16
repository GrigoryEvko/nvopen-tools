// Function: sub_1A2F700
// Address: 0x1a2f700
//
__int64 __fastcall sub_1A2F700(
        _BYTE **a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // r14
  unsigned __int64 v16; // rcx
  __int64 v17; // rsi
  unsigned int v18; // r14d
  _QWORD *v19; // rax
  __int64 v20; // rdi
  __int64 v21; // r11
  __int64 *v22; // rax
  double v23; // xmm4_8
  double v24; // xmm5_8
  __int64 v25; // rsi
  __int64 **v26; // rax
  _QWORD *v27; // rax
  __int64 *v28; // r10
  __int64 v29; // r13
  _BYTE *v30; // rdi
  unsigned __int64 v31; // r8
  _QWORD *v32; // rax
  double v33; // xmm4_8
  double v34; // xmm5_8
  double v35; // xmm4_8
  double v36; // xmm5_8
  __int64 v37; // rdx
  __int64 v38; // rcx
  double v39; // xmm4_8
  double v40; // xmm5_8
  _BYTE *v41; // rax
  __int64 v43; // r14
  __m128i v44; // rax
  __int16 v45; // r8
  __int64 v46; // r14
  _QWORD *v47; // r11
  unsigned __int8 v48; // r8
  unsigned int v49; // esi
  unsigned int v50; // eax
  __int16 v51; // si
  unsigned int v52; // r12d
  _QWORD *v53; // rax
  __int64 v54; // rsi
  unsigned int v55; // r14d
  unsigned int v56; // r14d
  _QWORD *v57; // rax
  __int64 *v58; // rax
  _BYTE *v59; // rdi
  __int64 **v60; // rcx
  __m128i v61; // rax
  __int64 v62; // rsi
  _QWORD *v63; // rax
  unsigned int v64; // r8d
  unsigned int v65; // eax
  __int16 v66; // si
  __int64 v67; // rax
  _BYTE *v68; // r9
  __int64 *v69; // rax
  __int64 v70; // rax
  _QWORD *v71; // rax
  _QWORD *v72; // rax
  _QWORD *v73; // [rsp+8h] [rbp-B8h]
  __int64 *v74; // [rsp+8h] [rbp-B8h]
  unsigned __int8 v75; // [rsp+8h] [rbp-B8h]
  _QWORD *v76; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v77; // [rsp+10h] [rbp-B0h]
  __int64 v78; // [rsp+10h] [rbp-B0h]
  unsigned int v79; // [rsp+10h] [rbp-B0h]
  _QWORD *v80; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v81; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v82; // [rsp+10h] [rbp-B0h]
  unsigned int v83; // [rsp+10h] [rbp-B0h]
  __int64 v84; // [rsp+10h] [rbp-B0h]
  _BYTE *v85; // [rsp+10h] [rbp-B0h]
  int v86; // [rsp+18h] [rbp-A8h]
  __int64 v87; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v88; // [rsp+28h] [rbp-98h]
  char v89; // [rsp+28h] [rbp-98h]
  __int64 *v90; // [rsp+28h] [rbp-98h]
  __int64 *v91; // [rsp+28h] [rbp-98h]
  __int64 v92; // [rsp+28h] [rbp-98h]
  __int64 *v93; // [rsp+28h] [rbp-98h]
  __int64 v94; // [rsp+30h] [rbp-90h] BYREF
  __int64 v95; // [rsp+38h] [rbp-88h]
  __int64 v96; // [rsp+40h] [rbp-80h]
  __m128i v97; // [rsp+50h] [rbp-70h] BYREF
  char v98; // [rsp+60h] [rbp-60h]
  char v99; // [rsp+61h] [rbp-5Fh]
  __m128i v100; // [rsp+70h] [rbp-50h] BYREF
  __int16 v101; // [rsp+80h] [rbp-40h]

  v12 = *(a2 - 3);
  v94 = 0;
  v95 = 0;
  v87 = v12;
  v96 = 0;
  sub_14A8180((__int64)a2, &v94, 0);
  v13 = *(_QWORD *)*(a2 - 3);
  if ( *(_BYTE *)(v13 + 8) == 16 )
    v13 = **(_QWORD **)(v13 + 16);
  v86 = *(_DWORD *)(v13 + 8) >> 8;
  if ( *((_BYTE *)a1 + 153) )
  {
    v52 = 8 * *((_DWORD *)a1 + 36);
    v53 = (_QWORD *)sub_16498A0((__int64)a2);
    v14 = sub_1644C60(v53, v52);
  }
  else
  {
    v14 = *a2;
  }
  v15 = sub_127FA20((__int64)*a1, v14);
  if ( a1[11] )
  {
    v16 = (unsigned __int64)a1[13];
    v17 = (__int64)a1[6];
    v88 = (a1[16] - a1[7]) / v16;
    v18 = (unsigned int)(1 << *(_WORD *)(v17 + 18)) >> 1;
    v77 = (a1[17] - a1[7]) / v16;
    v73 = sub_1A1D0C0((__int64 *)a1 + 24, v17, "load");
    sub_15F8F50((__int64)v73, v18);
    v101 = 259;
    v100.m128i_i64[0] = (__int64)"vec";
    v19 = sub_1A1CC60((__int64)(a1 + 24), (__int64)v73, v88, v77, &v100, (int)v73);
    v89 = 0;
    v20 = (__int64)*a1;
    v21 = (__int64)v19;
    goto LABEL_7;
  }
  if ( a1[10] && *(_BYTE *)(*a2 + 8) == 11 )
  {
    v54 = (__int64)a1[6];
    v55 = 1 << *(_WORD *)(v54 + 18);
    v90 = sub_1A1D0C0((__int64 *)a1 + 24, v54, "load");
    sub_15F8F50((__int64)v90, v55 >> 1);
    v21 = (__int64)sub_1A1C950((__int64)*a1, (__int64 *)a1 + 24, v90, (__int64)a1[10]);
    if ( a1[16] != a1[7] || a1[17] < a1[8] )
    {
      v81 = a1[16] - a1[7];
      v91 = (__int64 *)v21;
      v56 = 8 * *((_DWORD *)a1 + 36);
      v57 = (_QWORD *)sub_16498A0((__int64)a2);
      v58 = (__int64 *)sub_1644C60(v57, v56);
      v59 = *a1;
      v101 = 259;
      v100.m128i_i64[0] = (__int64)"extract";
      v21 = (__int64)sub_1A20950(v59, (__int64)(a1 + 24), v91, v58, v81, &v100, *(double *)a3.m128_u64, a4, a5);
    }
    v60 = (__int64 **)*a2;
    if ( *(_DWORD *)(*a2 + 8) >> 8 <= (unsigned __int64)(8LL * (_QWORD)a1[18]) )
      goto LABEL_37;
    v101 = 257;
    v71 = sub_1A1C8D0((__int64 *)a1 + 24, 37, v21, v60, &v100);
    v89 = 0;
    v20 = (__int64)*a1;
    v21 = (__int64)v71;
  }
  else if ( a1[16] == a1[7]
         && a1[17] == a1[8]
         && ((v82 = (unsigned __int64)a1[18], sub_1A1E350((__int64)*a1, (__int64)a1[9], v14))
          || v82 < (unsigned __int64)(v15 + 7) >> 3 && a1[9][8] == 11 && *(_BYTE *)(v14 + 8) == 11) )
  {
    v61.m128i_i64[0] = (__int64)sub_1649960((__int64)a2);
    v62 = (__int64)a1[6];
    v97 = v61;
    v101 = 261;
    v61.m128i_i16[4] = *((_WORD *)a2 + 9);
    v100.m128i_i64[0] = (__int64)&v97;
    v83 = (unsigned int)(1 << *(_WORD *)(v62 + 18)) >> 1;
    v63 = sub_1A1D230((__int64 *)a1 + 24, v62, v61.m128i_i8[8] & 1, &v100);
    v64 = v83;
    v84 = (__int64)v63;
    sub_15F8F50((__int64)v63, v64);
    v21 = v84;
    if ( v94 || v95 || v96 )
    {
      sub_1626170(v84, &v94);
      v21 = v84;
    }
    v65 = *((unsigned __int16 *)a2 + 9);
    if ( (v65 & 1) != 0 )
    {
      v66 = *(_WORD *)(v21 + 18);
      *(_BYTE *)(v21 + 56) = *((_BYTE *)a2 + 56);
      *(_WORD *)(v21 + 18) = v66 & 0x8000 | v66 & 0x7C7F | (((v65 >> 7) & 7) << 7);
    }
    if ( a2[6] || *((__int16 *)a2 + 9) < 0 )
    {
      v92 = v21;
      v67 = sub_1625790((__int64)a2, 11);
      v21 = v92;
      if ( v67 )
      {
        sub_1AEC950(a2, v67, v92);
        v21 = v92;
      }
    }
    v68 = a1[9];
    if ( v68[8] != 11 || *(_BYTE *)(v14 + 8) != 11 || *(_DWORD *)(v14 + 8) >> 8 <= *((_DWORD *)v68 + 2) >> 8 )
      goto LABEL_37;
    v85 = a1[9];
    v100.m128i_i64[0] = (__int64)"load.ext";
    v101 = 259;
    v69 = sub_1A1C8D0((__int64 *)a1 + 24, 37, v21, (__int64 **)v14, &v100);
    v20 = (__int64)*a1;
    v21 = (__int64)v69;
    v89 = **a1;
    if ( v89 )
    {
      v99 = 1;
      v97.m128i_i64[0] = (__int64)"endian_shift";
      v98 = 3;
      v93 = v69;
      v70 = sub_15A0680(*v69, (unsigned int)((*(_DWORD *)(v14 + 8) >> 8) - (*((_DWORD *)v85 + 2) >> 8)), 0);
      if ( *((_BYTE *)v93 + 16) > 0x10u || *(_BYTE *)(v70 + 16) > 0x10u )
      {
        v101 = 257;
        v72 = (_QWORD *)sub_15FB440(23, v93, v70, (__int64)&v100, 0);
        v21 = (__int64)sub_1A1C7B0((__int64 *)a1 + 24, v72, &v97);
      }
      else
      {
        v21 = sub_15A2D50(v93, v70, 0, 0, *(double *)a3.m128_u64, a4, a5);
      }
LABEL_37:
      v89 = 0;
      v20 = (__int64)*a1;
    }
  }
  else
  {
    v43 = sub_1647190((__int64 *)v14, v86);
    v44.m128i_i64[0] = (__int64)sub_1649960((__int64)a2);
    v45 = *((_WORD *)a2 + 9);
    v97 = v44;
    v101 = 261;
    v75 = v45 & 1;
    v100.m128i_i64[0] = (__int64)&v97;
    v79 = sub_1A22080((__int64 *)a1, v14);
    v46 = sub_1A246E0((__int64 *)a1, (__int64)(a1 + 24), v43);
    v47 = sub_1648A60(64, 1u);
    if ( v47 )
    {
      v48 = v75;
      v76 = v47;
      sub_15F9210((__int64)v47, *(_QWORD *)(*(_QWORD *)v46 + 24LL), v46, 0, v48, 0);
      v47 = v76;
    }
    v49 = v79;
    v80 = sub_1A1C7B0((__int64 *)a1 + 24, v47, &v100);
    sub_15F8F50((__int64)v80, v49);
    v21 = (__int64)v80;
    if ( v94 || v95 || v96 )
    {
      sub_1626170((__int64)v80, &v94);
      v21 = (__int64)v80;
    }
    v50 = *((unsigned __int16 *)a2 + 9);
    v89 = *((_WORD *)a2 + 9) & 1;
    if ( v89 )
    {
      v51 = *(_WORD *)(v21 + 18);
      *(_BYTE *)(v21 + 56) = *((_BYTE *)a2 + 56);
      *(_WORD *)(v21 + 18) = v51 & 0x8000 | v51 & 0x7C7F | (((v50 >> 7) & 7) << 7);
    }
    else
    {
      v89 = 1;
    }
    v20 = (__int64)*a1;
  }
LABEL_7:
  v22 = sub_1A1C950(v20, (__int64 *)a1 + 24, (__int64 *)v21, v14);
  if ( *((_BYTE *)a1 + 153) )
  {
    v25 = a2[4];
    v74 = v22;
    if ( v25 )
      v25 -= 24;
    sub_17050D0((__int64 *)a1 + 24, v25);
    v26 = (__int64 **)sub_1647190((__int64 *)*a2, v86);
    v78 = sub_1599EF0(v26);
    v27 = sub_1648A60(64, 1u);
    v28 = v74;
    v29 = (__int64)v27;
    if ( v27 )
    {
      sub_15F9210((__int64)v27, *(_QWORD *)(*(_QWORD *)v78 + 24LL), v78, 0, 0, 0);
      v28 = v74;
    }
    v30 = *a1;
    v31 = a1[16] - a1[14];
    v101 = 259;
    v100.m128i_i64[0] = (__int64)"insert";
    v32 = sub_1A202F0(v30, (__int64)(a1 + 24), v29, v28, v31, &v100, *(double *)a3.m128_u64, a4, a5);
    sub_164D160((__int64)a2, (__int64)v32, a3, a4, a5, a6, v33, v34, a9, a10);
    sub_164D160(v29, (__int64)a2, a3, a4, a5, a6, v35, v36, a9, a10);
    sub_164BEC0(v29, (__int64)a2, v37, v38, a3, a4, a5, a6, v39, v40, a9, a10);
  }
  else
  {
    sub_164D160((__int64)a2, (__int64)v22, a3, a4, a5, a6, v23, v24, a9, a10);
  }
  v41 = a1[4];
  v100.m128i_i64[0] = (__int64)a2;
  sub_1A2EDE0((__int64)(v41 + 208), v100.m128i_i64);
  v100.m128i_i64[0] = v87;
  if ( (unsigned __int8)sub_1AE9990(v87, 0) )
    sub_1A2EDE0((__int64)(a1[4] + 208), v100.m128i_i64);
  return (unsigned __int8)(v89 | *((_WORD *)a2 + 9) & 1) ^ 1u;
}
