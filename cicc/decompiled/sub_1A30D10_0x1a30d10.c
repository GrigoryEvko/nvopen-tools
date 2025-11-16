// Function: sub_1A30D10
// Address: 0x1a30d10
//
__int64 __fastcall sub_1A30D10(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v7; // rdx
  unsigned int v8; // eax
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  unsigned int v15; // r15d
  int v16; // eax
  __int64 v17; // rsi
  _QWORD *v18; // rdi
  int v19; // eax
  int v20; // eax
  __int64 v21; // r9
  __int64 v22; // r10
  __int64 v23; // rsi
  __int64 v24; // r8
  unsigned __int64 v25; // rdi
  __int64 *v26; // r13
  unsigned int v27; // r14d
  __int64 *v28; // rax
  __int64 *v29; // rdi
  __int64 v30; // r8
  const char *v31; // rax
  __int64 v32; // r9
  __int64 v33; // r8
  __int64 v34; // rdx
  __int64 *v35; // r14
  __int64 v36; // rax
  __int64 v37; // r10
  __int64 v38; // r11
  unsigned int v39; // eax
  __int64 v40; // rax
  unsigned __int8 v41; // al
  __int64 *v42; // r13
  __int64 v43; // r11
  __int64 v44; // rsi
  __int64 *v45; // rdi
  unsigned int v46; // ebx
  __int64 *v47; // rax
  __int64 v48; // rdi
  unsigned int v49; // ebx
  int v50; // eax
  bool v51; // cl
  _QWORD *v52; // r13
  __int64 v53; // rdi
  unsigned int v54; // ebx
  __int64 v56; // rsi
  __int64 v57; // rax
  __int64 v58; // r9
  __int64 v59; // rdx
  __int64 *v60; // r15
  _QWORD *v61; // r14
  _QWORD *v62; // r13
  __int64 *v63; // rbx
  unsigned __int8 v64; // al
  _QWORD *v65; // rdi
  __int64 v66; // rsi
  __int64 *v67; // rax
  unsigned int v68; // r13d
  _QWORD *v69; // rbx
  int v70; // r9d
  __int64 *v71; // rax
  __int64 v72; // r10
  __int64 *v73; // rax
  __int64 *v74; // rax
  __int64 v75; // rax
  __int64 v76; // r14
  __int64 *v77; // rax
  __int64 *v78; // rax
  unsigned int v79; // r8d
  __int64 *v80; // rax
  _BYTE *v81; // rdi
  unsigned __int64 v82; // r8
  __int64 *v83; // rax
  __int64 v84; // rsi
  __int64 *v85; // rax
  unsigned int v86; // r8d
  __int64 *v87; // rax
  _BYTE *v88; // rdi
  unsigned __int64 v89; // r8
  __int64 *v90; // rax
  __int64 *v91; // rax
  __int64 v92; // r13
  _QWORD *v93; // rax
  __int64 *v94; // rax
  __int64 v95; // rax
  int v96; // [rsp+0h] [rbp-100h]
  unsigned int v97; // [rsp+4h] [rbp-FCh]
  __int64 v98; // [rsp+8h] [rbp-F8h]
  __int64 v99; // [rsp+10h] [rbp-F0h]
  __int64 v100; // [rsp+18h] [rbp-E8h]
  int v101; // [rsp+28h] [rbp-D8h]
  unsigned int v102; // [rsp+28h] [rbp-D8h]
  unsigned int v103; // [rsp+2Ch] [rbp-D4h]
  __int64 v104; // [rsp+30h] [rbp-D0h]
  __int64 v105; // [rsp+38h] [rbp-C8h]
  __int64 v106; // [rsp+40h] [rbp-C0h]
  __int64 v107; // [rsp+40h] [rbp-C0h]
  __int64 v108; // [rsp+40h] [rbp-C0h]
  __int64 v109; // [rsp+40h] [rbp-C0h]
  __int64 v110; // [rsp+40h] [rbp-C0h]
  __int64 v111; // [rsp+40h] [rbp-C0h]
  unsigned __int8 v112; // [rsp+48h] [rbp-B8h]
  __int64 *v113; // [rsp+48h] [rbp-B8h]
  __int64 v114; // [rsp+48h] [rbp-B8h]
  __int64 v115; // [rsp+48h] [rbp-B8h]
  unsigned int v116; // [rsp+48h] [rbp-B8h]
  __int64 *v117; // [rsp+48h] [rbp-B8h]
  unsigned int v118; // [rsp+48h] [rbp-B8h]
  __int64 *v119; // [rsp+48h] [rbp-B8h]
  unsigned __int64 v120; // [rsp+50h] [rbp-B0h] BYREF
  unsigned int v121; // [rsp+58h] [rbp-A8h]
  const char *v122; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v123; // [rsp+68h] [rbp-98h]
  __int64 v124; // [rsp+70h] [rbp-90h] BYREF
  __int64 v125; // [rsp+78h] [rbp-88h]
  __int64 v126; // [rsp+80h] [rbp-80h]
  __m128i v127; // [rsp+90h] [rbp-70h] BYREF
  __int64 v128; // [rsp+A0h] [rbp-60h]
  __int128 v129; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v130; // [rsp+C0h] [rbp-40h]

  v126 = 0;
  v124 = 0;
  v125 = 0;
  sub_14A8180(a2, &v124, 0);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v104 = *(_QWORD *)(a2 - 8);
  else
    v104 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v7 = *(_QWORD *)(a1 + 48);
  v105 = *(_QWORD *)(a1 + 160);
  v8 = (unsigned int)(1 << *(_WORD *)(v7 + 18)) >> 1;
  if ( !v8 )
    v8 = sub_15A9FE0(*(_QWORD *)a1, *(_QWORD *)(v7 + 56));
  v9 = *(_QWORD *)(a1 + 56);
  v10 = (*(_QWORD *)(a1 + 128) - v9) | v8;
  v99 = v10 & -v10;
  v103 = v10 & -(int)v10;
  v112 = *(_BYTE *)(a1 + 152);
  if ( !v112 )
  {
    v66 = sub_1A246E0((__int64 *)a1, a1 + 192, **(_QWORD **)(a1 + 168));
    if ( v105 == v104 )
    {
      sub_1593B40((_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), v66);
      *(_QWORD *)&v129 = *(_QWORD *)(a2 + 56);
      v73 = (__int64 *)sub_16498A0(a2);
      *(_QWORD *)(a2 + 56) = sub_1563C10((__int64 *)&v129, v73, 1, 1);
      if ( !(_DWORD)v99 )
        goto LABEL_80;
      v74 = (__int64 *)sub_16498A0(a2);
      v75 = sub_155D330(v74, (unsigned int)v99);
      v127.m128i_i32[0] = 0;
      v76 = v75;
    }
    else
    {
      sub_1593B40((_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))), v66);
      *(_QWORD *)&v129 = *(_QWORD *)(a2 + 56);
      v67 = (__int64 *)sub_16498A0(a2);
      *(_QWORD *)(a2 + 56) = sub_1563C10((__int64 *)&v129, v67, 2, 1);
      if ( !(_DWORD)v99 )
        goto LABEL_80;
      v94 = (__int64 *)sub_16498A0(a2);
      v95 = sub_155D330(v94, (unsigned int)v99);
      v127.m128i_i32[0] = 1;
      v76 = v95;
    }
    *(_QWORD *)&v129 = *(_QWORD *)(a2 + 56);
    v77 = (__int64 *)sub_16498A0(a2);
    *(_QWORD *)(a2 + 56) = sub_1563E10((__int64 *)&v129, v77, v127.m128i_i32, 1, v76);
LABEL_80:
    *(_QWORD *)&v129 = *(_QWORD *)(a1 + 168);
    if ( (unsigned __int8)sub_1AE9990(v129, 0) )
      sub_1A2EDE0(*(_QWORD *)(a1 + 32) + 208LL, (__int64 *)&v129);
    return v112;
  }
  if ( *(_QWORD *)(a1 + 88)
    || *(_QWORD *)(a1 + 80)
    || *(_QWORD *)(a1 + 112) <= v9
    && *(_QWORD *)(a1 + 120) >= *(_QWORD *)(a1 + 64)
    && (v92 = *(_QWORD *)(a1 + 144),
        v92 == (unsigned __int64)(sub_127FA20(*(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 56LL)) + 7) >> 3) )
  {
    v112 = 0;
    goto LABEL_8;
  }
  if ( *(_QWORD *)(a1 + 40) != *(_QWORD *)(a1 + 48) )
  {
LABEL_8:
    v11 = *(_QWORD *)(a1 + 32);
    *(_QWORD *)&v129 = a2;
    sub_1A2EDE0(v11 + 208, (__int64 *)&v129);
    if ( v105 == v104 )
      v106 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    else
      v106 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v12 = sub_164A820(v106);
    if ( *(_BYTE *)(v12 + 16) == 53 )
    {
      *(_QWORD *)&v129 = v12;
      sub_1A30BB0(*(_QWORD *)(a1 + 32) + 32LL, (__int64 *)&v129);
    }
    v13 = *(_QWORD *)v106;
    v14 = *(_QWORD *)v106;
    if ( *(_BYTE *)(*(_QWORD *)v106 + 8LL) == 16 )
      v14 = **(_QWORD **)(v13 + 16);
    v15 = *(_DWORD *)(v14 + 8) >> 8;
    v16 = sub_15A9520(*(_QWORD *)a1, v15);
    v17 = *(_QWORD *)(a1 + 128) - *(_QWORD *)(a1 + 112);
    v121 = 8 * v16;
    if ( (unsigned int)(8 * v16) <= 0x40 )
      v120 = v17 & (0xFFFFFFFFFFFFFFFFLL >> (-8 * (unsigned __int8)v16));
    else
      sub_16A4EF0((__int64)&v120, v17, 0);
    v18 = (_QWORD *)(a2 + 56);
    if ( v105 == v104 )
      v19 = sub_15603A0(v18, 1);
    else
      v19 = sub_15603A0(v18, 0);
    v101 = v19;
    sub_16A5D10((__int64)&v129, (__int64)&v120, 0x40u);
    v20 = v101;
    if ( !v101 )
      v20 = 1;
    if ( DWORD2(v129) <= 0x40 )
    {
      v102 = (v129 | v20) & -(v129 | v20);
      if ( !v112 )
      {
LABEL_22:
        v21 = *(_QWORD *)(a1 + 128);
        v22 = *(_QWORD *)(a1 + 56);
        v23 = *(_QWORD *)(a1 + 136);
        if ( v21 == v22 )
          v112 = *(_QWORD *)(a1 + 64) == v23;
        v24 = *(_QWORD *)(a1 + 88);
        if ( v24 )
        {
          v25 = *(_QWORD *)(a1 + 104);
          v26 = *(__int64 **)(a1 + 80);
          v97 = (v21 - v22) / v25;
          v96 = (v23 - v22) / v25;
          v27 = v96 - v97;
          if ( !v26 )
          {
LABEL_27:
            if ( !v112 )
            {
              v29 = *(__int64 **)(v24 + 24);
              if ( v27 != 1 )
                v29 = sub_16463B0(v29, v27);
              goto LABEL_30;
            }
LABEL_65:
            v29 = *(__int64 **)(a1 + 72);
LABEL_30:
            v30 = sub_1647190(v29, v15);
            goto LABEL_31;
          }
        }
        else
        {
          v97 = 0;
          v26 = *(__int64 **)(a1 + 80);
          if ( !v26 )
          {
            v96 = 0;
            goto LABEL_65;
          }
          v96 = 0;
          v27 = 0;
        }
        v28 = (__int64 *)sub_1644C60((_QWORD *)*v26, 8 * ((int)v23 - (int)v21));
        v24 = *(_QWORD *)(a1 + 88);
        v26 = v28;
        if ( !v24 )
        {
          if ( *(_QWORD *)(a1 + 80) && !v112 )
          {
            v30 = sub_1647190(v28, v15);
LABEL_31:
            v98 = v30;
            v31 = sub_1649960(v106);
            v33 = v98;
            v122 = v31;
            LOWORD(v130) = 773;
            *(_QWORD *)&v129 = &v122;
            *((_QWORD *)&v129 + 1) = ".";
            v123 = v34;
            v127.m128i_i32[2] = v121;
            if ( v121 > 0x40 )
            {
              sub_16A4FD0((__int64)&v127, (const void **)&v120);
              v33 = v98;
            }
            else
            {
              v127.m128i_i64[0] = v120;
            }
            v35 = (__int64 *)(a1 + 192);
            v36 = sub_1A23B30(a1 + 192, *(_QWORD *)a1, v106, (__int64)&v127, v33, v32, v129, v130);
            v37 = v36;
            if ( v127.m128i_i32[2] > 0x40u && v127.m128i_i64[0] )
            {
              v107 = v36;
              j_j___libc_free_0_0(v127.m128i_i64[0]);
              v37 = v107;
            }
            v38 = *(_QWORD *)(a1 + 48);
            if ( v105 == v104 )
              goto LABEL_40;
            if ( *(_QWORD *)(a1 + 88) )
            {
              if ( v112 )
              {
LABEL_39:
                v39 = v102;
                v102 = v99;
                v103 = v39;
                v40 = v37;
                v37 = *(_QWORD *)(a1 + 48);
                v38 = v40;
LABEL_40:
                v100 = v38;
                v108 = v37;
                v127.m128i_i64[0] = (__int64)"copyload";
                LOWORD(v128) = 259;
                v41 = sub_1A211D0(a2);
                v42 = sub_1A1D230((__int64 *)(a1 + 192), v108, v41, &v127);
                sub_15F8F50((__int64)v42, v102);
                v43 = v100;
                if ( v124 || v125 || v126 )
                {
                  sub_1626170((__int64)v42, &v124);
                  v43 = v100;
                }
                if ( *(_QWORD *)(a1 + 88) )
                {
                  if ( v112 != 1 && v105 == v104 )
                  {
                    v44 = *(_QWORD *)(a1 + 48);
                    v45 = (__int64 *)(a1 + 192);
                    v109 = v43;
                    v46 = 1 << *(_WORD *)(v44 + 18);
                    v113 = sub_1A1D0C0(v45, v44, "oldload");
                    sub_15F8F50((__int64)v113, v46 >> 1);
                    v127.m128i_i64[0] = (__int64)"vec";
                    LOWORD(v128) = 259;
                    v47 = sub_1A1DB70((__int64)v35, v113, (__int64)v42, v97, &v127, (int)v113);
                    v43 = v109;
                    v42 = v47;
                  }
                }
                else if ( *(_QWORD *)(a1 + 80) && v112 != 1 && v105 == v104 )
                {
                  v84 = *(_QWORD *)(a1 + 48);
                  v111 = v43;
                  v118 = (unsigned int)(1 << *(_WORD *)(v84 + 18)) >> 1;
                  v85 = sub_1A1D0C0((__int64 *)(a1 + 192), v84, "oldload");
                  v86 = v118;
                  v119 = v85;
                  sub_15F8F50((__int64)v85, v86);
                  v87 = sub_1A1C950(*(_QWORD *)a1, (__int64 *)(a1 + 192), v119, *(_QWORD *)(a1 + 80));
                  v88 = *(_BYTE **)a1;
                  v89 = *(_QWORD *)(a1 + 128) - *(_QWORD *)(a1 + 56);
                  v127.m128i_i64[0] = (__int64)"insert";
                  LOWORD(v128) = 259;
                  v90 = sub_1A202F0(v88, a1 + 192, (__int64)v87, v42, v89, &v127, a3, a4, a5);
                  v91 = sub_1A1C950(*(_QWORD *)a1, (__int64 *)(a1 + 192), v90, *(_QWORD *)(a1 + 72));
                  v43 = v111;
                  v42 = v91;
                }
                goto LABEL_46;
              }
              v115 = v37;
              v68 = 1 << *(_WORD *)(v38 + 18);
              v69 = sub_1A1D0C0((__int64 *)(a1 + 192), v38, "load");
              sub_15F8F50((__int64)v69, v68 >> 1);
              v127.m128i_i64[0] = (__int64)"vec";
              LOWORD(v128) = 259;
              v71 = sub_1A1CC60((__int64)v35, (__int64)v69, v97, v96, &v127, v70);
              v72 = v115;
              v42 = v71;
            }
            else
            {
              if ( !*(_QWORD *)(a1 + 80) || v112 )
                goto LABEL_39;
              v110 = v37;
              v116 = (unsigned int)(1 << *(_WORD *)(v38 + 18)) >> 1;
              v78 = sub_1A1D0C0((__int64 *)(a1 + 192), v38, "load");
              v79 = v116;
              v117 = v78;
              sub_15F8F50((__int64)v78, v79);
              v80 = sub_1A1C950(*(_QWORD *)a1, (__int64 *)(a1 + 192), v117, *(_QWORD *)(a1 + 80));
              v81 = *(_BYTE **)a1;
              v82 = *(_QWORD *)(a1 + 128) - *(_QWORD *)(a1 + 56);
              v127.m128i_i64[0] = (__int64)"extract";
              LOWORD(v128) = 259;
              v83 = sub_1A20950(v81, a1 + 192, v80, v26, v82, &v127, a3, a4, a5);
              v72 = v110;
              v42 = v83;
            }
            v43 = v72;
            v103 = v102;
LABEL_46:
            v48 = *(_QWORD *)(a2 + 24 * (3LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
            v49 = *(_DWORD *)(v48 + 32);
            if ( v49 <= 0x40 )
            {
              v51 = *(_QWORD *)(v48 + 24) == 0;
            }
            else
            {
              v114 = v43;
              v50 = sub_16A57B0(v48 + 24);
              v43 = v114;
              v51 = v49 == v50;
            }
            v52 = sub_1A1CF60(v35, (__int64)v42, v43, !v51);
            sub_15F9450((__int64)v52, v103);
            if ( v124 || v125 || v126 )
              sub_1626170((__int64)v52, &v124);
            v53 = *(_QWORD *)(a2 + 24 * (3LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
            v54 = *(_DWORD *)(v53 + 32);
            if ( v54 <= 0x40 )
              v112 = *(_QWORD *)(v53 + 24) == 0;
            else
              v112 = v54 == (unsigned int)sub_16A57B0(v53 + 24);
LABEL_52:
            if ( v121 > 0x40 && v120 )
              j_j___libc_free_0_0(v120);
            return v112;
          }
          goto LABEL_65;
        }
        goto LABEL_27;
      }
    }
    else
    {
      v102 = (*(_DWORD *)v129 | v20) & -(*(_DWORD *)v129 | v20);
      j_j___libc_free_0_0(v129);
      if ( !v112 )
        goto LABEL_22;
    }
    v122 = sub_1649960(v106);
    v127.m128i_i64[0] = (__int64)&v122;
    v127.m128i_i64[1] = (__int64)".";
    v123 = v59;
    LOWORD(v128) = 773;
    DWORD2(v129) = v121;
    if ( v121 > 0x40 )
      sub_16A4FD0((__int64)&v129, (const void **)&v120);
    else
      *(_QWORD *)&v129 = v120;
    v60 = (__int64 *)(a1 + 192);
    v61 = (_QWORD *)sub_1A23B30(a1 + 192, *(_QWORD *)a1, v106, (__int64)&v129, v13, v58, *(_OWORD *)&v127, v128);
    sub_135E100((__int64 *)&v129);
    v62 = (_QWORD *)sub_1A246E0((__int64 *)a1, a1 + 192, **(_QWORD **)(a1 + 168));
    v63 = (__int64 *)sub_15A0680(
                       **(_QWORD **)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
                       *(_QWORD *)(a1 + 136) - *(_QWORD *)(a1 + 128),
                       0);
    if ( v105 == v104 )
    {
      v103 = v102;
      v93 = v61;
      v61 = v62;
      v102 = v99;
      v62 = v93;
    }
    v64 = sub_1A211D0(a2);
    v65 = sub_15E7430(v60, v61, v102, v62, v103, v63, v64, 0, 0, 0, 0);
    if ( v124 || v125 || (v112 = 0, v126) )
    {
      sub_1626170((__int64)v65, &v124);
      v112 = 0;
    }
    goto LABEL_52;
  }
  v112 = 0;
  v56 = *(_QWORD *)(a1 + 136);
  if ( v56 != *(_QWORD *)(a1 + 120) )
  {
    v57 = sub_15A0680(
            **(_QWORD **)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
            v56 - *(_QWORD *)(a1 + 128),
            0);
    sub_1593B40((_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))), v57);
  }
  return v112;
}
