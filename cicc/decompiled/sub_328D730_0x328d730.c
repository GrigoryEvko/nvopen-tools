// Function: sub_328D730
// Address: 0x328d730
//
__int64 __fastcall sub_328D730(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // r13
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r14
  char v11; // al
  __int64 v12; // rdx
  const __m128i *v13; // rax
  __int64 *v14; // r8
  __int32 v15; // edi
  __m128i v16; // xmm0
  int v17; // esi
  __int64 v18; // rcx
  __int64 v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rax
  unsigned __int16 *v22; // rax
  __int64 v23; // rax
  __int16 v24; // cx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rbx
  __int64 v28; // r14
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rcx
  int v33; // esi
  int v34; // ebx
  bool v35; // al
  int v36; // r9d
  int v37; // r15d
  __int128 v38; // rax
  __int64 v39; // rbx
  __int128 v40; // rax
  int v41; // r9d
  unsigned __int64 v42; // rbx
  int v43; // r9d
  bool v44; // al
  __int64 v45; // r14
  int v46; // esi
  __int64 v47; // rax
  __int64 v48; // rdx
  int v49; // esi
  bool v50; // al
  char v51; // al
  __int64 v52; // rdx
  char v53; // al
  __int64 v54; // rdx
  __m128i v55; // xmm1
  __int64 v56; // rax
  __int64 v57; // rdx
  int v58; // eax
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  int v62; // r14d
  __int128 v63; // rax
  int v64; // r9d
  __int128 v65; // rax
  __int64 v66; // rsi
  __int64 v67; // rdx
  __int64 v68; // rcx
  __int64 v69; // r8
  __int128 v70; // rax
  int v71; // r9d
  __int64 v72; // rdx
  __int64 v73; // rbx
  __int64 v74; // rcx
  __int128 v75; // rax
  int v76; // r9d
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rax
  __int64 *v80; // rbx
  __int64 *v81; // rsi
  __int64 v82; // rax
  __int64 v83; // rdx
  __int64 v84; // r15
  __int64 v85; // r14
  __int128 v86; // rax
  int v87; // r9d
  __int64 v88; // rax
  __int64 v89; // rbx
  int v90; // r13d
  __int64 v91; // rdx
  __int64 v92; // r15
  __int64 v93; // r14
  __int128 v94; // rax
  int v95; // r9d
  __int128 v96; // [rsp-30h] [rbp-160h]
  __int128 v97; // [rsp-20h] [rbp-150h]
  __int128 v98; // [rsp+0h] [rbp-130h]
  __int64 v99; // [rsp+10h] [rbp-120h]
  __int64 v100; // [rsp+18h] [rbp-118h]
  __int16 v101; // [rsp+22h] [rbp-10Eh]
  unsigned int v102; // [rsp+24h] [rbp-10Ch]
  unsigned int v103; // [rsp+28h] [rbp-108h]
  __m128i v104; // [rsp+30h] [rbp-100h] BYREF
  __int64 v105; // [rsp+40h] [rbp-F0h]
  __int64 v106; // [rsp+48h] [rbp-E8h]
  unsigned int *v107; // [rsp+50h] [rbp-E0h]
  __int64 v108; // [rsp+58h] [rbp-D8h]
  int v109; // [rsp+60h] [rbp-D0h]
  unsigned __int32 v110; // [rsp+64h] [rbp-CCh]
  char v111; // [rsp+69h] [rbp-C7h]
  char v112; // [rsp+6Ah] [rbp-C6h]
  bool v113; // [rsp+6Bh] [rbp-C5h]
  unsigned int v114; // [rsp+6Ch] [rbp-C4h]
  __int64 *v115; // [rsp+70h] [rbp-C0h]
  _QWORD *v116; // [rsp+78h] [rbp-B8h]
  _QWORD *v117; // [rsp+80h] [rbp-B0h]
  __int64 v118; // [rsp+88h] [rbp-A8h]
  __int128 v119; // [rsp+90h] [rbp-A0h]
  __int128 v120; // [rsp+A0h] [rbp-90h]
  unsigned int v121; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v122; // [rsp+B8h] [rbp-78h]
  __int64 v123; // [rsp+C0h] [rbp-70h] BYREF
  int v124; // [rsp+C8h] [rbp-68h]
  __int64 v125; // [rsp+D0h] [rbp-60h] BYREF
  int v126; // [rsp+D8h] [rbp-58h]
  __int64 v127; // [rsp+E0h] [rbp-50h] BYREF
  int v128; // [rsp+E8h] [rbp-48h]
  __m128i v129; // [rsp+F0h] [rbp-40h] BYREF

  v2 = *(_QWORD **)(a1 + 40);
  v3 = *v2;
  if ( *(_DWORD *)(*v2 + 24LL) != 208 )
    return 0;
  v5 = v2[5];
  if ( *(_DWORD *)(v5 + 24) != 208 )
    return 0;
  v6 = *(_QWORD *)(v3 + 56);
  if ( !v6 )
    return 0;
  if ( *(_QWORD *)(v6 + 32) )
    return 0;
  v7 = *(_QWORD *)(v5 + 56);
  if ( !v7 || *(_QWORD *)(v7 + 32) )
    return 0;
  v8 = *(_QWORD *)(a2 + 16);
  v11 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v8 + 2176LL))(v8, a1, v3, v5);
  v12 = *(_QWORD *)(v5 + 40);
  v112 = v11;
  v13 = *(const __m128i **)(v3 + 40);
  v14 = *(__int64 **)(v12 + 40);
  v15 = v13->m128i_i32[2];
  v16 = _mm_loadu_si128(v13);
  v108 = *(_QWORD *)v12;
  v17 = *(_DWORD *)(v12 + 8);
  v18 = v13->m128i_i64[0];
  v110 = v15;
  v109 = v17;
  v19 = v13[2].m128i_i64[1];
  v20 = v13[3].m128i_i64[0];
  LODWORD(v13) = v13[3].m128i_i32[0];
  v116 = (_QWORD *)v18;
  *(_QWORD *)&v119 = v19;
  DWORD2(v119) = (_DWORD)v13;
  v21 = *(_QWORD *)(v12 + 48);
  LODWORD(v12) = *(_DWORD *)(v12 + 48);
  v115 = v14;
  v117 = v14;
  LODWORD(v118) = v12;
  v104 = v16;
  *(_QWORD *)&v120 = v21;
  v100 = sub_33DFBC0(v19, v20, 0, 0);
  v99 = sub_33DFBC0(v115, v120, 0, 0);
  v114 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v3 + 40) + 80LL) + 96LL);
  v102 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v5 + 40) + 80LL) + 96LL);
  v22 = *(unsigned __int16 **)(a1 + 48);
  v105 = *((_QWORD *)v22 + 1);
  v103 = *v22;
  v106 = v110;
  v23 = v116[6] + 16LL * v110;
  v24 = *(_WORD *)v23;
  v122 = *(_QWORD *)(v23 + 8);
  v101 = v24;
  v123 = *(_QWORD *)(a1 + 80);
  LOWORD(v121) = v24;
  v115 = &v123;
  if ( v123 )
    sub_325F5D0(&v123);
  v124 = *(_DWORD *)(a1 + 72);
  LODWORD(v120) = v121;
  v113 = sub_328D6E0(v8, 0x11Au, v121);
  if ( v113 )
    v113 = sub_328D6E0(v8, 0x119u, v120);
  v111 = sub_328A020(v8, 0x118u, v120, v122, 0);
  if ( v111 )
    v111 = sub_328A020(v8, 0x117u, v120, v122, 0);
  v107 = &v121;
  if ( !sub_3280180((__int64)&v121)
    || !sub_328D6E0(v8, 0xB7u, v120)
    || !sub_328D6E0(v8, 0xB5u, v120)
    || !sub_328D6E0(v8, 0xB6u, v120)
    || !sub_328D6E0(v8, 0xB4u, v120) )
  {
    if ( v101 )
    {
      if ( (unsigned __int16)(v101 - 10) > 6u
        && (unsigned __int16)(v101 - 126) > 0x31u
        && (unsigned __int16)(v101 - 208) > 0x14u )
      {
        goto LABEL_25;
      }
    }
    else if ( !(unsigned __int8)sub_3007030((__int64)v107) )
    {
      goto LABEL_25;
    }
    if ( !v113 && !v111 )
      goto LABEL_25;
  }
  if ( v114 <= 0x16 && ((0x42C3C3uLL >> v114) & 1) != 0 )
    goto LABEL_25;
  if ( v114 == v102 )
  {
    v120 = 0u;
    if ( v109 == v110 && (_QWORD *)v108 == v116 )
    {
      *(_QWORD *)&v120 = v116;
      *((_QWORD *)&v120 + 1) = v106 | *((_QWORD *)&v120 + 1) & 0xFFFFFFFF00000000LL;
      v42 = (unsigned int)sub_33CBD20(v114);
      goto LABEL_49;
    }
    if ( (_DWORD)v118 != DWORD2(v119) || v117 != (_QWORD *)v119 )
      goto LABEL_25;
    *(_QWORD *)&v120 = v119;
    *((_QWORD *)&v120 + 1) = DWORD2(v119) | *((_QWORD *)&v120 + 1) & 0xFFFFFFFF00000000LL;
    LODWORD(v118) = v109;
    v117 = (_QWORD *)v108;
    goto LABEL_60;
  }
  if ( (unsigned int)sub_33CBD20(v102) == v114 )
  {
    v120 = 0u;
    if ( v117 == v116 && (_DWORD)v118 == v110 )
    {
      v42 = v102;
      *(_QWORD *)&v120 = v116;
      *((_QWORD *)&v120 + 1) = v106 | *((_QWORD *)&v120 + 1) & 0xFFFFFFFF00000000LL;
      LODWORD(v118) = v109;
      v117 = (_QWORD *)v108;
LABEL_49:
      switch ( (_DWORD)v42 )
      {
        case 0x14:
          if ( (unsigned __int8)sub_33E0720(v120, *((_QWORD *)&v120 + 1), 0) )
            goto LABEL_25;
          break;
        case 0x12:
          if ( (unsigned __int8)sub_33E07E0(v120, *((_QWORD *)&v120 + 1), 0) )
            goto LABEL_25;
          break;
        case 0x18:
          goto LABEL_25;
        default:
          LODWORD(v106) = v42 - 18;
          v44 = sub_3280180((__int64)v107);
          if ( v44 )
          {
            if ( (((_DWORD)v42 - 13) & 0xFFFFFFF7) == 0 )
            {
LABEL_54:
              if ( v44 == (*(_DWORD *)(a1 + 24) == 187) )
              {
                v45 = (unsigned int)v118;
                if ( (unsigned int)v106 <= 3 )
                  v46 = 180;
                else
                  v46 = 182;
                *((_QWORD *)&v119 + 1) = DWORD2(v119);
              }
              else
              {
                v45 = (unsigned int)v118;
                if ( (unsigned int)v106 <= 3 )
                  v46 = 181;
                else
                  v46 = 183;
                *((_QWORD *)&v119 + 1) = DWORD2(v119);
              }
              goto LABEL_58;
            }
LABEL_53:
            v44 = (((_DWORD)v42 - 12) & 0xFFFFFFF7) == 0;
            goto LABEL_54;
          }
LABEL_65:
          if ( !(unsigned __int8)sub_3280140((__int64)v107) )
            goto LABEL_25;
          v49 = *(_DWORD *)(a1 + 24);
          v45 = (unsigned int)v118;
          *((_QWORD *)&v119 + 1) = DWORD2(v119);
          v50 = (unsigned int)(v42 - 20) <= 1 && v49 == 187;
          if ( !v50 )
          {
            if ( (unsigned int)v106 > 1 )
            {
              v50 = v49 == 186;
              if ( (unsigned int)(v42 - 20) > 1 || v49 != 186 )
              {
                if ( ((unsigned int)(v42 - 4) > 1 || v49 != 187) && ((unsigned int)(v42 - 10) > 1 || v49 != 186) )
                {
LABEL_74:
                  if ( ((unsigned int)(v42 - 2) > 1 || v49 != 187) && ((unsigned int)(v42 - 12) > 1 || !v50) )
                    goto LABEL_25;
                  v46 = 280;
                  if ( v111 )
                  {
LABEL_58:
                    *((_QWORD *)&v98 + 1) = v45;
                    *(_QWORD *)&v98 = v117;
                    v47 = sub_3406EB0(a2, v46, (_DWORD)v115, v121, v122, v43, v119, v98);
                    v25 = sub_32889F0(a2, (int)v115, v103, v105, v47, v48, v120, v42, 0);
                    goto LABEL_28;
                  }
                  v51 = sub_325DA90(v119, DWORD2(v119), (__int64)v117, (unsigned int)v118, a2);
                  if ( v113 && v51 )
                  {
LABEL_81:
                    v46 = 282;
                    goto LABEL_58;
                  }
LABEL_25:
                  if ( !v112
                    || v114 != v102
                    || v114 != 5 * (*(_DWORD *)(a1 + 24) == 186) + 17
                    || v108 != (_QWORD)v116
                    || v109 != v110
                    || !v100
                    || !v99
                    || !sub_3280180((__int64)v107) )
                  {
                    goto LABEL_27;
                  }
                  v27 = *(_QWORD *)(v100 + 96);
                  *(_QWORD *)&v120 = v27 + 24;
                  v28 = *(_QWORD *)(v99 + 96) + 24LL;
                  *((_QWORD *)&v119 + 1) = &v125;
                  sub_9865C0((__int64)&v125, v28);
                  sub_AADAA0((__int64)&v127, (__int64)&v125, v29, v30, v31);
                  if ( sub_AAD8B0(v27 + 24, &v127) )
                  {
                    if ( (v112 & 4) != 0
                      || (v55 = _mm_load_si128(&v104),
                          *(_QWORD *)&v119 = &v129,
                          v129 = v55,
                          v56 = sub_33ED250(a2, v121, v122, v32),
                          (unsigned __int8)sub_33CEDC0(a2, 189, v56, v57, v119, 1)) )
                    {
                      sub_969240(&v127);
                      sub_969240(*((__int64 **)&v119 + 1));
                      v33 = *(_DWORD *)(v27 + 32);
                      v34 = v120;
                      v35 = sub_986C60((__int64 *)v120, v33 - 1);
                      v37 = (int)v115;
                      if ( !v35 )
                        LODWORD(v28) = v34;
                      *(_QWORD *)&v38 = sub_33FAF80(a2, 189, (_DWORD)v115, v121, v122, v36, *(_OWORD *)&v104);
                      v39 = *(_QWORD *)(v3 + 40);
                      v120 = v38;
                      *(_QWORD *)&v40 = sub_34007B0(a2, v28, v37, v121, v122, 0, 0);
                      v25 = sub_340F900(a2, 208, v37, v103, v105, v41, v120, v40, *(_OWORD *)(v39 + 80));
                      goto LABEL_28;
                    }
                  }
                  sub_969240(&v127);
                  sub_969240(*((__int64 **)&v119 + 1));
                  if ( (v112 & 3) == 0 )
                    goto LABEL_27;
                  v58 = sub_C4C880(v28, v120);
                  if ( v58 <= 0 )
                  {
                    if ( v58 )
                    {
                      v79 = v120;
                      *(_QWORD *)&v120 = v28;
                      v28 = v79;
                    }
                    else
                    {
                      v28 = v120;
                    }
                  }
                  sub_9865C0((__int64)&v129, v28);
                  sub_C46B40((__int64)&v129, (__int64 *)v120);
                  v126 = v129.m128i_i32[2];
                  v125 = v129.m128i_i64[0];
                  if ( sub_9867B0(*((__int64 *)&v119 + 1)) || !sub_986BA0(*((__int64 *)&v119 + 1)) )
                    goto LABEL_117;
                  if ( sub_986760(v28) && (v112 & 2) != 0 )
                  {
                    v80 = v115;
                    v81 = v115;
                    v82 = sub_34074A0(a2, v115, v104.m128i_i64[0], v104.m128i_i64[1], v121, v122);
                    v84 = v83;
                    v85 = v82;
                    *(_QWORD *)&v86 = sub_34007B0(a2, v120, (_DWORD)v80, v121, v122, 0, 0);
                    *((_QWORD *)&v97 + 1) = v84;
                    *(_QWORD *)&v97 = v85;
                    v115 = v80;
                    v88 = sub_3406EB0(a2, 186, (_DWORD)v80, v121, v122, v87, v97, v86);
                    v89 = *(_QWORD *)(v3 + 40);
                    v90 = (int)v115;
                    v92 = v91;
                    v93 = v88;
                    *(_QWORD *)&v94 = sub_3400BD0(a2, 0, (_DWORD)v115, v121, v122, 0, 0, v81);
                    *((_QWORD *)&v96 + 1) = v92;
                    *(_QWORD *)&v96 = v93;
                    v77 = sub_340F900(a2, 208, v90, v103, v105, v95, v96, v94, *(_OWORD *)(v89 + 80));
                  }
                  else
                  {
                    if ( (v112 & 1) == 0 )
                    {
LABEL_117:
                      sub_969240(*((__int64 **)&v119 + 1));
                      goto LABEL_27;
                    }
                    sub_9865C0((__int64)&v127, v120);
                    sub_AADAA0((__int64)&v129, (__int64)&v127, v59, v60, v61);
                    v62 = (int)v115;
                    *(_QWORD *)&v63 = sub_34007B0(a2, (unsigned int)&v129, (_DWORD)v115, v121, v122, 0, 0);
                    *(_QWORD *)&v65 = sub_3406EB0(a2, 56, v62, v121, v122, v64, *(_OWORD *)&v104, v63);
                    v120 = v65;
                    sub_969240(v129.m128i_i64);
                    sub_969240(&v127);
                    v66 = *((_QWORD *)&v119 + 1);
                    sub_9865C0((__int64)&v127, *((__int64 *)&v119 + 1));
                    sub_987160((__int64)&v127, v66, v67, v68, v69);
                    v129.m128i_i32[2] = v128;
                    v128 = 0;
                    v129.m128i_i64[0] = v127;
                    *(_QWORD *)&v70 = sub_34007B0(a2, (unsigned int)&v129, v62, v121, v122, 0, 0);
                    *(_QWORD *)&v120 = sub_3406EB0(a2, 186, v62, v121, v122, v71, v120, v70);
                    *((_QWORD *)&v120 + 1) = v72;
                    sub_969240(v129.m128i_i64);
                    sub_969240(&v127);
                    v73 = *(_QWORD *)(v3 + 40);
                    *(_QWORD *)&v75 = sub_3400BD0(a2, 0, v62, v121, v122, 0, 0, v74);
                    v77 = sub_340F900(a2, 208, v62, v103, v105, v76, v120, v75, *(_OWORD *)(v73 + 80));
                  }
                  *(_QWORD *)&v119 = v78;
                  *(_QWORD *)&v120 = v77;
                  sub_969240(*((__int64 **)&v119 + 1));
                  v26 = v119;
                  v25 = v120;
                  goto LABEL_28;
                }
                v46 = 279;
                if ( v111 )
                  goto LABEL_58;
                v53 = sub_325DA90(v119, DWORD2(v119), (__int64)v117, (unsigned int)v118, a2);
                if ( !v113 || !v53 )
                  goto LABEL_25;
LABEL_91:
                v46 = 281;
                goto LABEL_58;
              }
LABEL_84:
              v52 = (unsigned int)v118;
              v118 = *((_QWORD *)&v119 + 1);
              if ( (unsigned __int8)sub_33CE830(a2, v117, v52, 0, 0)
                && (unsigned __int8)sub_33CE830(a2, v119, v118, 0, 0)
                && v113 )
              {
                goto LABEL_81;
              }
              goto LABEL_25;
            }
            if ( v49 != 186 )
            {
              if ( v49 != 187 )
                goto LABEL_74;
              goto LABEL_84;
            }
          }
          v54 = (unsigned int)v118;
          v118 = *((_QWORD *)&v119 + 1);
          if ( !(unsigned __int8)sub_33CE830(a2, v117, v54, 0, 0)
            || !(unsigned __int8)sub_33CE830(a2, v119, v118, 0, 0)
            || !v113 )
          {
            goto LABEL_25;
          }
          goto LABEL_91;
      }
      LODWORD(v106) = v42 - 18;
      if ( sub_3280180((__int64)v107) )
        goto LABEL_53;
      goto LABEL_65;
    }
    if ( v108 != (_QWORD)v119 || DWORD2(v119) != v109 )
      goto LABEL_25;
    *(_QWORD *)&v120 = v119;
    *((_QWORD *)&v120 + 1) = DWORD2(v119) | *((_QWORD *)&v120 + 1) & 0xFFFFFFFF00000000LL;
LABEL_60:
    v42 = v114;
    DWORD2(v119) = v110;
    *(_QWORD *)&v119 = v116;
    goto LABEL_49;
  }
LABEL_27:
  v25 = 0;
  v26 = 0;
LABEL_28:
  *((_QWORD *)&v119 + 1) = v26;
  *(_QWORD *)&v120 = v25;
  sub_9C6650(v115);
  return v120;
}
