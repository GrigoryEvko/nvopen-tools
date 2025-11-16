// Function: sub_18B96C0
// Address: 0x18b96c0
//
__int64 __fastcall sub_18B96C0(
        __int64 a1,
        __int64 a2,
        _BYTE *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  const __m128i *v14; // rbx
  __int64 **v16; // rax
  __int64 v17; // r10
  _QWORD *v18; // rax
  _QWORD *v19; // r15
  __int64 *v20; // r13
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // r9
  double v24; // xmm4_8
  double v25; // xmm5_8
  __int64 v26; // rsi
  unsigned __int8 *v27; // rsi
  _QWORD *v28; // r13
  _QWORD *v29; // rdi
  __m128 v30; // xmm0
  __int64 v31; // r13
  __int64 v32; // rax
  unsigned __int8 *v33; // rsi
  __int64 v34; // rax
  __int64 v35; // r9
  __int64 **v36; // rdx
  _QWORD *v37; // rax
  __int64 v38; // r15
  __int64 *v39; // r13
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rsi
  unsigned __int8 *v43; // rsi
  unsigned __int8 v44; // al
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // r9
  _QWORD *v48; // r13
  double v49; // xmm4_8
  double v50; // xmm5_8
  __int64 v51; // r15
  _QWORD *v52; // rdi
  __int64 v53; // rax
  __int64 v54; // r9
  __int64 *v55; // r15
  __int64 v56; // rsi
  __int64 v57; // rax
  __int64 v58; // rsi
  __int64 v59; // rdx
  unsigned __int8 *v60; // rsi
  _QWORD *v61; // rax
  __int64 v62; // r9
  _QWORD **v63; // rax
  __int64 *v64; // rax
  __int64 v65; // rax
  __int64 v66; // r9
  __int64 v67; // rsi
  __int64 v68; // rax
  __int64 v69; // rsi
  unsigned __int8 *v70; // rsi
  __int64 v71; // rax
  __int64 *v72; // r13
  __int64 v73; // rax
  __int64 v74; // rcx
  __int64 v75; // rsi
  unsigned __int8 *v76; // rsi
  __int64 v77; // rax
  __int64 v78; // r10
  __int64 *v79; // r15
  __int64 v80; // rsi
  __int64 v81; // rax
  __int64 v82; // rsi
  __int64 v83; // rdx
  unsigned __int8 *v84; // rsi
  __int64 result; // rax
  unsigned int v86; // r13d
  __int64 v87; // [rsp+8h] [rbp-168h]
  __int64 v88; // [rsp+10h] [rbp-160h]
  _QWORD *v89; // [rsp+10h] [rbp-160h]
  __int64 *v90; // [rsp+10h] [rbp-160h]
  __int64 v91; // [rsp+10h] [rbp-160h]
  const __m128i *v97; // [rsp+40h] [rbp-130h]
  __int64 v98; // [rsp+48h] [rbp-128h]
  __int64 v99; // [rsp+48h] [rbp-128h]
  __int64 v100; // [rsp+48h] [rbp-128h]
  __int64 v101; // [rsp+48h] [rbp-128h]
  __int64 v102; // [rsp+48h] [rbp-128h]
  __int64 v103; // [rsp+48h] [rbp-128h]
  __int64 v104; // [rsp+48h] [rbp-128h]
  __int64 v105; // [rsp+48h] [rbp-128h]
  __int64 v106; // [rsp+48h] [rbp-128h]
  __int64 v107; // [rsp+48h] [rbp-128h]
  __int64 v108; // [rsp+48h] [rbp-128h]
  __int64 v109; // [rsp+48h] [rbp-128h]
  unsigned __int8 *v110; // [rsp+68h] [rbp-108h] BYREF
  __m128 v111; // [rsp+70h] [rbp-100h] BYREF
  _DWORD *v112; // [rsp+80h] [rbp-F0h]
  __int64 v113[2]; // [rsp+90h] [rbp-E0h] BYREF
  __int16 v114; // [rsp+A0h] [rbp-D0h]
  __int64 v115[2]; // [rsp+B0h] [rbp-C0h] BYREF
  __int16 v116; // [rsp+C0h] [rbp-B0h]
  unsigned __int8 *v117[2]; // [rsp+D0h] [rbp-A0h] BYREF
  __int16 v118; // [rsp+E0h] [rbp-90h]
  unsigned __int8 *v119; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v120; // [rsp+F8h] [rbp-78h]
  unsigned __int64 v121; // [rsp+100h] [rbp-70h]
  __int64 v122; // [rsp+108h] [rbp-68h]
  __int64 v123; // [rsp+110h] [rbp-60h]
  int v124; // [rsp+118h] [rbp-58h]
  __int64 v125; // [rsp+120h] [rbp-50h]
  __int64 v126; // [rsp+128h] [rbp-48h]

  v14 = *(const __m128i **)a2;
  v97 = *(const __m128i **)(a2 + 8);
  if ( *(const __m128i **)a2 != v97 )
  {
    do
    {
      v30 = (__m128)_mm_loadu_si128(v14);
      v111 = v30;
      v112 = (_DWORD *)v14[1].m128i_i64[0];
      v31 = *(_QWORD *)(v30.m128_u64[1] & 0xFFFFFFFFFFFFFFF8LL);
      v32 = sub_16498A0(v30.m128_u64[1] & 0xFFFFFFFFFFFFFFF8LL);
      v125 = 0;
      v126 = 0;
      v33 = *(unsigned __int8 **)((v30.m128_u64[1] & 0xFFFFFFFFFFFFFFF8LL) + 48);
      v122 = v32;
      v124 = 0;
      v34 = *(_QWORD *)((v30.m128_u64[1] & 0xFFFFFFFFFFFFFFF8LL) + 40);
      v119 = 0;
      v120 = v34;
      v123 = 0;
      v121 = (v30.m128_u64[1] & 0xFFFFFFFFFFFFFFF8LL) + 24;
      v117[0] = v33;
      if ( v33 )
      {
        sub_1623A60((__int64)v117, (__int64)v33, 2);
        if ( v119 )
          sub_161E7C0((__int64)&v119, (__int64)v119);
        v119 = v117[0];
        if ( v117[0] )
          sub_1623210((__int64)v117, v117[0], (__int64)&v119);
      }
      v35 = v111.m128_u64[0];
      v114 = 257;
      v36 = *(__int64 ***)(a1 + 48);
      v116 = 257;
      if ( v36 != *(__int64 ***)v111.m128_u64[0] )
      {
        if ( *(_BYTE *)(v111.m128_u64[0] + 16) > 0x10u )
        {
          v118 = 257;
          v53 = sub_15FDBD0(47, v111.m128_i64[0], (__int64)v36, (__int64)v117, 0);
          v54 = v53;
          if ( v120 )
          {
            v55 = (__int64 *)v121;
            v101 = v53;
            sub_157E9D0(v120 + 40, v53);
            v54 = v101;
            v56 = *v55;
            v57 = *(_QWORD *)(v101 + 24);
            *(_QWORD *)(v101 + 32) = v55;
            v56 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v101 + 24) = v56 | v57 & 7;
            *(_QWORD *)(v56 + 8) = v101 + 24;
            *v55 = *v55 & 7 | (v101 + 24);
          }
          v102 = v54;
          sub_164B780(v54, v113);
          v35 = v102;
          if ( v119 )
          {
            v110 = v119;
            sub_1623A60((__int64)&v110, (__int64)v119, 2);
            v35 = v102;
            v58 = *(_QWORD *)(v102 + 48);
            v59 = v102 + 48;
            if ( v58 )
            {
              v88 = v102;
              v103 = v102 + 48;
              sub_161E7C0(v103, v58);
              v35 = v88;
              v59 = v103;
            }
            v60 = v110;
            *(_QWORD *)(v35 + 48) = v110;
            if ( v60 )
            {
              v104 = v35;
              sub_1623210((__int64)&v110, v60, v59);
              v35 = v104;
            }
          }
        }
        else
        {
          v35 = sub_15A46C0(47, (__int64 ***)v111.m128_u64[0], v36, 0);
        }
      }
      v100 = sub_12815B0((__int64 *)&v119, *(_QWORD *)(a1 + 40), (_BYTE *)v35, a5, (__int64)v115);
      if ( *(_DWORD *)(v31 + 8) >> 8 != 1 )
      {
        v116 = 257;
        v16 = (__int64 **)sub_1647190((__int64 *)v31, 0);
        v17 = v100;
        if ( v16 != *(__int64 ***)v100 )
        {
          if ( *(_BYTE *)(v100 + 16) > 0x10u )
          {
            v118 = 257;
            v77 = sub_15FDBD0(47, v100, (__int64)v16, (__int64)v117, 0);
            v78 = v77;
            if ( v120 )
            {
              v79 = (__int64 *)v121;
              v107 = v77;
              sub_157E9D0(v120 + 40, v77);
              v78 = v107;
              v80 = *v79;
              v81 = *(_QWORD *)(v107 + 24);
              *(_QWORD *)(v107 + 32) = v79;
              v80 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v107 + 24) = v80 | v81 & 7;
              *(_QWORD *)(v80 + 8) = v107 + 24;
              *v79 = *v79 & 7 | (v107 + 24);
            }
            v108 = v78;
            sub_164B780(v78, v115);
            v17 = v108;
            if ( v119 )
            {
              v113[0] = (__int64)v119;
              sub_1623A60((__int64)v113, (__int64)v119, 2);
              v17 = v108;
              v82 = *(_QWORD *)(v108 + 48);
              v83 = v108 + 48;
              if ( v82 )
              {
                sub_161E7C0(v108 + 48, v82);
                v17 = v108;
                v83 = v108 + 48;
              }
              v84 = (unsigned __int8 *)v113[0];
              *(_QWORD *)(v17 + 48) = v113[0];
              if ( v84 )
              {
                v109 = v17;
                sub_1623210((__int64)v113, v84, v83);
                v17 = v109;
              }
            }
          }
          else
          {
            v17 = sub_15A46C0(47, (__int64 ***)v100, v16, 0);
          }
        }
        v98 = v17;
        v118 = 257;
        v18 = sub_1648A60(64, 1u);
        v19 = v18;
        if ( v18 )
          sub_15F9210((__int64)v18, v31, v98, 0, 0, 0);
        if ( v120 )
        {
          v20 = (__int64 *)v121;
          sub_157E9D0(v120 + 40, (__int64)v19);
          v21 = v19[3];
          v22 = *v20;
          v19[4] = v20;
          v22 &= 0xFFFFFFFFFFFFFFF8LL;
          v19[3] = v22 | v21 & 7;
          *(_QWORD *)(v22 + 8) = v19 + 3;
          *v20 = *v20 & 7 | (unsigned __int64)(v19 + 3);
        }
        sub_164B780((__int64)v19, (__int64 *)v117);
        if ( v119 )
        {
          v115[0] = (__int64)v119;
          sub_1623A60((__int64)v115, (__int64)v119, 2);
          v26 = v19[6];
          if ( v26 )
            sub_161E7C0((__int64)(v19 + 6), v26);
          v27 = (unsigned __int8 *)v115[0];
          v19[6] = v115[0];
          if ( v27 )
            sub_1623210((__int64)v115, v27, (__int64)(v19 + 6));
        }
        if ( *(_BYTE *)(a1 + 80) )
          sub_18B6C20(
            (__int64)&v111,
            "virtual-const-prop",
            18,
            a3,
            a4,
            v23,
            *(__int64 (__fastcall **)(__int64, __int64))(a1 + 88),
            *(_QWORD *)(a1 + 96));
        sub_164D160(v111.m128_u64[1] & 0xFFFFFFFFFFFFFFF8LL, (__int64)v19, v30, a8, a9, a10, v24, v25, a13, a14);
        v28 = (_QWORD *)(v111.m128_u64[1] & 0xFFFFFFFFFFFFFFF8LL);
        if ( *(_BYTE *)((v111.m128_u64[1] & 0xFFFFFFFFFFFFFFF8LL) + 16) != 29 )
          goto LABEL_21;
        v99 = *(v28 - 6);
        v29 = sub_1648A60(56, 1u);
        if ( v29 )
          sub_15F8320((__int64)v29, v99, (__int64)v28);
        goto LABEL_20;
      }
      v118 = 257;
      v37 = sub_1648A60(64, 1u);
      v38 = (__int64)v37;
      if ( v37 )
        sub_15F9210((__int64)v37, *(_QWORD *)(*(_QWORD *)v100 + 24LL), v100, 0, 0, 0);
      if ( v120 )
      {
        v39 = (__int64 *)v121;
        sub_157E9D0(v120 + 40, v38);
        v40 = *(_QWORD *)(v38 + 24);
        v41 = *v39;
        *(_QWORD *)(v38 + 32) = v39;
        v41 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v38 + 24) = v41 | v40 & 7;
        *(_QWORD *)(v41 + 8) = v38 + 24;
        *v39 = *v39 & 7 | (v38 + 24);
      }
      sub_164B780(v38, (__int64 *)v117);
      if ( v119 )
      {
        v115[0] = (__int64)v119;
        sub_1623A60((__int64)v115, (__int64)v119, 2);
        v42 = *(_QWORD *)(v38 + 48);
        if ( v42 )
          sub_161E7C0(v38 + 48, v42);
        v43 = (unsigned __int8 *)v115[0];
        *(_QWORD *)(v38 + 48) = v115[0];
        if ( v43 )
          sub_1623210((__int64)v115, v43, v38 + 48);
      }
      v116 = 257;
      v44 = *(_BYTE *)(a6 + 16);
      if ( v44 > 0x10u )
      {
LABEL_74:
        v118 = 257;
        v71 = sub_15FB440(26, (__int64 *)v38, a6, (__int64)v117, 0);
        v38 = v71;
        if ( v120 )
        {
          v72 = (__int64 *)v121;
          sub_157E9D0(v120 + 40, v71);
          v73 = *(_QWORD *)(v38 + 24);
          v74 = *v72;
          *(_QWORD *)(v38 + 32) = v72;
          v74 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v38 + 24) = v74 | v73 & 7;
          *(_QWORD *)(v74 + 8) = v38 + 24;
          *v72 = *v72 & 7 | (v38 + 24);
        }
        sub_164B780(v38, v115);
        if ( v119 )
        {
          v113[0] = (__int64)v119;
          sub_1623A60((__int64)v113, (__int64)v119, 2);
          v75 = *(_QWORD *)(v38 + 48);
          if ( v75 )
            sub_161E7C0(v38 + 48, v75);
          v76 = (unsigned __int8 *)v113[0];
          *(_QWORD *)(v38 + 48) = v113[0];
          if ( v76 )
            sub_1623210((__int64)v113, v76, v38 + 48);
        }
        goto LABEL_48;
      }
      if ( v44 == 13 )
      {
        v86 = *(_DWORD *)(a6 + 32);
        if ( v86 <= 0x40 )
        {
          if ( *(_QWORD *)(a6 + 24) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v86) )
            goto LABEL_48;
        }
        else if ( v86 == (unsigned int)sub_16A58F0(a6 + 24) )
        {
          goto LABEL_48;
        }
      }
      if ( *(_BYTE *)(v38 + 16) > 0x10u )
        goto LABEL_74;
      v38 = sub_15A2CF0((__int64 *)v38, a6, *(double *)v30.m128_u64, a8, a9);
LABEL_48:
      v45 = *(_QWORD *)(a1 + 40);
      v116 = 257;
      v46 = sub_159C470(v45, 0, 0);
      if ( *(_BYTE *)(v38 + 16) > 0x10u || *(_BYTE *)(v46 + 16) > 0x10u )
      {
        v105 = v46;
        v118 = 257;
        v61 = sub_1648A60(56, 2u);
        v62 = v105;
        v48 = v61;
        if ( v61 )
        {
          v106 = (__int64)v61;
          v63 = *(_QWORD ***)v38;
          if ( *(_BYTE *)(*(_QWORD *)v38 + 8LL) == 16 )
          {
            v87 = v62;
            v89 = v63[4];
            v64 = (__int64 *)sub_1643320(*v63);
            v65 = (__int64)sub_16463B0(v64, (unsigned int)v89);
            v66 = v87;
          }
          else
          {
            v91 = v62;
            v65 = sub_1643320(*v63);
            v66 = v91;
          }
          sub_15FEC10((__int64)v48, v65, 51, 33, v38, v66, (__int64)v117, 0);
        }
        else
        {
          v106 = 0;
        }
        if ( v120 )
        {
          v90 = (__int64 *)v121;
          sub_157E9D0(v120 + 40, (__int64)v48);
          v67 = *v90;
          v68 = v48[3] & 7LL;
          v48[4] = v90;
          v67 &= 0xFFFFFFFFFFFFFFF8LL;
          v48[3] = v67 | v68;
          *(_QWORD *)(v67 + 8) = v48 + 3;
          *v90 = *v90 & 7 | (unsigned __int64)(v48 + 3);
        }
        sub_164B780(v106, v115);
        if ( v119 )
        {
          v113[0] = (__int64)v119;
          sub_1623A60((__int64)v113, (__int64)v119, 2);
          v69 = v48[6];
          if ( v69 )
            sub_161E7C0((__int64)(v48 + 6), v69);
          v70 = (unsigned __int8 *)v113[0];
          v48[6] = v113[0];
          if ( v70 )
            sub_1623210((__int64)v113, v70, (__int64)(v48 + 6));
        }
      }
      else
      {
        v48 = (_QWORD *)sub_15A37B0(0x21u, (_QWORD *)v38, (_QWORD *)v46, 0);
      }
      if ( *(_BYTE *)(a1 + 80) )
        sub_18B6C20(
          (__int64)&v111,
          "virtual-const-prop-1-bit",
          24,
          a3,
          a4,
          v47,
          *(__int64 (__fastcall **)(__int64, __int64))(a1 + 88),
          *(_QWORD *)(a1 + 96));
      sub_164D160(v111.m128_u64[1] & 0xFFFFFFFFFFFFFFF8LL, (__int64)v48, v30, a8, a9, a10, v49, v50, a13, a14);
      v28 = (_QWORD *)(v111.m128_u64[1] & 0xFFFFFFFFFFFFFFF8LL);
      if ( *(_BYTE *)((v111.m128_u64[1] & 0xFFFFFFFFFFFFFFF8LL) + 16) != 29 )
        goto LABEL_21;
      v51 = *(v28 - 6);
      v52 = sub_1648A60(56, 1u);
      if ( v52 )
        sub_15F8320((__int64)v52, v51, (__int64)v28);
LABEL_20:
      sub_157F2D0(*(v28 - 3), v28[5], 0);
      v28 = (_QWORD *)(v111.m128_u64[1] & 0xFFFFFFFFFFFFFFF8LL);
LABEL_21:
      sub_15F20C0(v28);
      if ( v112 )
        --*v112;
      if ( v119 )
        sub_161E7C0((__int64)&v119, (__int64)v119);
      v14 = (const __m128i *)((char *)v14 + 24);
    }
    while ( v97 != v14 );
  }
  *(_BYTE *)(a2 + 24) = 1;
  result = *(_QWORD *)(a2 + 32);
  if ( result != *(_QWORD *)(a2 + 40) )
    *(_QWORD *)(a2 + 40) = result;
  return result;
}
