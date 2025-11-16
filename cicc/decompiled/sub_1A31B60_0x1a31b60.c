// Function: sub_1A31B60
// Address: 0x1a31b60
//
__int64 __fastcall sub_1A31B60(__int64 *a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v7; // rax
  __int64 *v8; // r10
  __int64 v9; // rsi
  unsigned __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 *v15; // r10
  __int64 v16; // rax
  __m128i v17; // xmm0
  __int64 v18; // rdi
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rbx
  unsigned __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rsi
  unsigned int v24; // r14d
  __int16 v25; // cx
  unsigned int v26; // r15d
  _QWORD *v27; // rax
  unsigned int v28; // r15d
  _QWORD *v29; // rbx
  __int64 v30; // rax
  unsigned __int64 *v31; // rcx
  __m128i v32; // xmm1
  unsigned __int64 v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rsi
  __int64 v37; // rdx
  unsigned __int8 *v38; // rsi
  __int64 v39; // rdi
  __int64 result; // rax
  __int64 v41; // rsi
  __int64 v42; // rax
  __int64 *v43; // r10
  __int64 v44; // r14
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rax
  unsigned int v48; // r14d
  _QWORD *v49; // rbx
  unsigned int v50; // eax
  __int64 v51; // rax
  unsigned int v52; // r14d
  _QWORD *v53; // rax
  __int64 *v54; // rax
  _BYTE *v55; // rdi
  unsigned __int64 v56; // r8
  __int64 v57; // rax
  __int16 v58; // si
  _QWORD *v59; // rax
  char v60; // al
  __int64 v61; // r11
  __int64 *v62; // rax
  __int64 *v63; // rax
  __int64 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // rax
  _QWORD *v67; // rax
  __int64 *v68; // rax
  __int64 v69; // [rsp+0h] [rbp-F0h]
  __int16 v70; // [rsp+8h] [rbp-E8h]
  __int64 v71; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v72; // [rsp+8h] [rbp-E8h]
  __int64 v73; // [rsp+8h] [rbp-E8h]
  __int64 v74; // [rsp+10h] [rbp-E0h]
  __int64 v75; // [rsp+10h] [rbp-E0h]
  __int64 v76; // [rsp+10h] [rbp-E0h]
  unsigned __int64 *v77; // [rsp+10h] [rbp-E0h]
  __int64 *v78; // [rsp+18h] [rbp-D8h]
  __int64 *v79; // [rsp+18h] [rbp-D8h]
  __int64 v80; // [rsp+18h] [rbp-D8h]
  unsigned __int64 *v81; // [rsp+18h] [rbp-D8h]
  __int64 *v82; // [rsp+18h] [rbp-D8h]
  __int64 *v83; // [rsp+18h] [rbp-D8h]
  __int64 v84; // [rsp+18h] [rbp-D8h]
  __int64 *v85; // [rsp+18h] [rbp-D8h]
  unsigned int v86; // [rsp+18h] [rbp-D8h]
  __m128i v87; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v88; // [rsp+30h] [rbp-C0h]
  __m128i v89; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v90; // [rsp+50h] [rbp-A0h]
  __m128i v91; // [rsp+60h] [rbp-90h] BYREF
  __int64 v92; // [rsp+70h] [rbp-80h]
  __m128i v93; // [rsp+80h] [rbp-70h] BYREF
  __int64 v94; // [rsp+90h] [rbp-60h]
  __m128i v95; // [rsp+A0h] [rbp-50h] BYREF
  __int16 v96; // [rsp+B0h] [rbp-40h]

  v7 = *(_QWORD *)(a2 - 24);
  v87 = 0u;
  v74 = v7;
  v88 = 0;
  sub_14A8180(a2, v87.m128i_i64, 0);
  v8 = *(__int64 **)(a2 - 48);
  v9 = *v8;
  if ( *(_BYTE *)(*v8 + 8) == 15 )
  {
    v84 = *(_QWORD *)(a2 - 48);
    v57 = sub_164A820(v84);
    v8 = (__int64 *)v84;
    if ( *(_BYTE *)(v57 + 16) == 53 )
    {
      v95.m128i_i64[0] = v57;
      sub_1A30BB0(a1[4] + 320, v95.m128i_i64);
      v8 = (__int64 *)v84;
    }
    v9 = *v8;
  }
  v78 = v8;
  v10 = a1[18];
  v11 = sub_127FA20(*a1, v9);
  v15 = v78;
  if ( v10 < (unsigned __int64)(v11 + 7) >> 3 )
  {
    v52 = 8 * *((_DWORD *)a1 + 36);
    v53 = (_QWORD *)sub_16498A0(a2);
    v54 = (__int64 *)sub_1644C60(v53, v52);
    v55 = (_BYTE *)*a1;
    v56 = a1[16] - a1[14];
    v95.m128i_i64[0] = (__int64)"extract";
    v96 = 259;
    v15 = sub_1A20950(v55, (__int64)(a1 + 24), v78, v54, v56, &v95, a3, a4, a5);
  }
  v16 = a1[11];
  if ( v16 )
  {
    v17 = _mm_loadu_si128(&v87);
    v90 = v88;
    v89 = v17;
    v18 = *v15;
    if ( v16 != *v15 )
    {
      v19 = a1[13];
      v20 = (a1[16] - a1[7]) / v19;
      v21 = (a1[17] - a1[7]) / v19;
      v22 = a1[12];
      if ( (_DWORD)v21 - (_DWORD)v20 != 1 )
      {
        v85 = v15;
        v59 = sub_16463B0((__int64 *)a1[12], (int)v21 - (int)v20);
        v15 = v85;
        v22 = (__int64)v59;
        v18 = *v85;
      }
      if ( v22 != v18 )
        v15 = sub_1A1C950(*a1, a1 + 24, v15, v22);
      v23 = a1[6];
      v75 = (__int64)v15;
      v24 = 1 << *(_WORD *)(v23 + 18);
      v79 = sub_1A1D0C0(a1 + 24, v23, "load");
      sub_15F8F50((__int64)v79, v24 >> 1);
      v95.m128i_i64[0] = (__int64)"vec";
      v96 = 259;
      v15 = sub_1A1DB70((__int64)(a1 + 24), v79, v75, v20, &v95, (int)v79);
    }
    v76 = (__int64)v15;
    v25 = *(_WORD *)(a1[6] + 18);
    v80 = a1[6];
    LOWORD(v92) = 257;
    v26 = 1 << v25;
    v27 = sub_1648A60(64, 2u);
    v28 = v26 >> 1;
    v29 = v27;
    if ( v27 )
      sub_15F9650((__int64)v27, v76, v80, 0, 0);
    v30 = a1[25];
    v31 = (unsigned __int64 *)a1[26];
    if ( (unsigned __int8)v92 > 1u )
    {
      v71 = a1[25];
      v95.m128i_i64[0] = (__int64)(a1 + 32);
      v77 = v31;
      v96 = 260;
      sub_14EC200(&v93, &v95, &v91);
      v31 = v77;
      v30 = v71;
    }
    else
    {
      v32 = _mm_loadu_si128(&v91);
      v94 = v92;
      v93 = v32;
    }
    v81 = v31;
    if ( v30 )
    {
      sub_157E9D0(v30 + 40, (__int64)v29);
      v33 = *v81;
      v34 = v29[3] & 7LL;
      v29[4] = v81;
      v33 &= 0xFFFFFFFFFFFFFFF8LL;
      v29[3] = v33 | v34;
      *(_QWORD *)(v33 + 8) = v29 + 3;
      *v81 = *v81 & 7 | (unsigned __int64)(v29 + 3);
    }
    sub_164B780((__int64)v29, v93.m128i_i64);
    v35 = a1[24];
    if ( v35 )
    {
      v95.m128i_i64[0] = a1[24];
      sub_1623A60((__int64)&v95, v35, 2);
      v36 = v29[6];
      v37 = (__int64)(v29 + 6);
      if ( v36 )
      {
        sub_161E7C0((__int64)(v29 + 6), v36);
        v37 = (__int64)(v29 + 6);
      }
      v38 = (unsigned __int8 *)v95.m128i_i64[0];
      v29[6] = v95.m128i_i64[0];
      if ( v38 )
        sub_1623210((__int64)&v95, v38, v37);
    }
    sub_15F9450((__int64)v29, v28);
    if ( v89.m128i_i64[0] || __PAIR128__(v89.m128i_u64[1], 0) != v90 )
      sub_1626170((__int64)v29, v89.m128i_i64);
    v39 = a1[4];
    v95.m128i_i64[0] = a2;
    sub_1A2EDE0(v39 + 208, v95.m128i_i64);
    return 1;
  }
  v41 = *v15;
  if ( a1[10] && *(_BYTE *)(v41 + 8) == 11 )
    return sub_1A2F3F0(a1, v15, a2, a3, a4, a5, v12, v13, v14, v87.m128i_i64[0], v87.m128i_i64[1], v88);
  v82 = v15;
  v42 = sub_127FA20(*a1, v41);
  v43 = v82;
  v44 = v42;
  v45 = *v82;
  if ( a1[16] != a1[7] || a1[17] != a1[8] )
    goto LABEL_30;
  v72 = a1[18];
  v60 = sub_1A1E350(*a1, *v82, a1[9]);
  v43 = v82;
  v45 = *v82;
  if ( !v60 )
  {
    if ( v72 >= (unsigned __int64)(v44 + 7) >> 3
      || (v61 = a1[9], *(_BYTE *)(v61 + 8) != 11)
      || *(_BYTE *)(v45 + 8) != 11 )
    {
LABEL_30:
      v46 = **(_QWORD **)(a2 - 24);
      if ( *(_BYTE *)(v46 + 8) == 16 )
        v46 = **(_QWORD **)(v46 + 16);
      v83 = v43;
      v47 = sub_1647190((__int64 *)v45, *(_DWORD *)(v46 + 8) >> 8);
      v69 = sub_1A246E0(a1, (__int64)(a1 + 24), v47);
      v70 = *(_WORD *)(a2 + 18) & 1;
      v48 = sub_1A22080(a1, *v83);
      v49 = sub_1A1CF60(a1 + 24, (__int64)v83, v69, v70);
      sub_15F9450((__int64)v49, v48);
      goto LABEL_33;
    }
    goto LABEL_54;
  }
  v61 = a1[9];
  if ( *(_BYTE *)(v45 + 8) == 11 && *(_BYTE *)(v61 + 8) == 11 )
  {
LABEL_54:
    if ( *(_DWORD *)(v61 + 8) >> 8 < *(_DWORD *)(v45 + 8) >> 8 )
    {
      if ( *(_BYTE *)*a1 )
      {
        v93.m128i_i64[0] = (__int64)"endian_shift";
        LOWORD(v94) = 259;
        v73 = v61;
        v65 = sub_15A0680(*v82, (unsigned int)((*(_DWORD *)(v45 + 8) >> 8) - (*(_DWORD *)(v61 + 8) >> 8)), 0);
        if ( *((_BYTE *)v82 + 16) > 0x10u || *(_BYTE *)(v65 + 16) > 0x10u )
        {
          v96 = 257;
          v67 = (_QWORD *)sub_15FB440(24, v82, v65, (__int64)&v95, 0);
          v68 = sub_1A1C7B0(a1 + 24, v67, &v93);
          v61 = v73;
          v43 = v68;
        }
        else
        {
          v66 = sub_15A2D80(v82, v65, 0, a3, a4, a5);
          v61 = v73;
          v43 = (__int64 *)v66;
        }
      }
      v95.m128i_i64[0] = (__int64)"load.trunc";
      v96 = 259;
      v62 = sub_1A1C8D0(a1 + 24, 36, (__int64)v43, (__int64 **)v61, &v95);
      v61 = a1[9];
      v43 = v62;
    }
  }
  v63 = sub_1A1C950(*a1, a1 + 24, v43, v61);
  v64 = a1[6];
  v86 = (unsigned int)(1 << *(_WORD *)(v64 + 18)) >> 1;
  v49 = sub_1A1CF60(a1 + 24, (__int64)v63, v64, *(_BYTE *)(a2 + 18) & 1);
  sub_15F9450((__int64)v49, v86);
LABEL_33:
  v95.m128i_i32[0] = 10;
  sub_15F4370((__int64)v49, a2, v95.m128i_i32, 1);
  if ( v87.m128i_i64[0] || __PAIR128__(v87.m128i_u64[1], 0) != v88 )
    sub_1626170((__int64)v49, v87.m128i_i64);
  v50 = *(unsigned __int16 *)(a2 + 18);
  if ( (v50 & 1) != 0 )
  {
    v58 = *((_WORD *)v49 + 9);
    *((_BYTE *)v49 + 56) = *(_BYTE *)(a2 + 56);
    *((_WORD *)v49 + 9) = v58 & 0x8000 | v58 & 0x7C7F | (((v50 >> 7) & 7) << 7);
  }
  v51 = a1[4];
  v95.m128i_i64[0] = a2;
  sub_1A2EDE0(v51 + 208, v95.m128i_i64);
  v95.m128i_i64[0] = v74;
  if ( (unsigned __int8)sub_1AE9990(v74, 0) )
    sub_1A2EDE0(a1[4] + 208, v95.m128i_i64);
  result = 0;
  if ( a1[6] == *(v49 - 3) )
    return !(*(_WORD *)(a2 + 18) & 1);
  return result;
}
