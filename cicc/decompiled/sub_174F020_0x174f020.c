// Function: sub_174F020
// Address: 0x174f020
//
__int64 __fastcall sub_174F020(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v12; // r14
  char v13; // r13
  _BYTE *v16; // rdi
  unsigned __int8 v17; // al
  __int64 v18; // rdi
  int v19; // eax
  __int64 *v20; // r11
  __int64 **v21; // r15
  unsigned int v22; // edx
  __int64 v23; // r11
  bool v24; // al
  bool v25; // al
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // r9
  int v29; // eax
  __int64 v30; // r11
  __int64 v31; // rdi
  unsigned __int8 *v32; // r13
  __int64 v33; // r8
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned int v37; // esi
  __int64 v39; // rax
  __int64 v40; // r13
  int v41; // eax
  __int64 v42; // rbx
  _QWORD *v43; // rax
  double v44; // xmm4_8
  double v45; // xmm5_8
  __int64 v46; // r15
  unsigned int v48; // r15d
  bool v49; // al
  __int64 *v50; // r13
  __int64 v51; // r15
  int v52; // eax
  __int64 v53; // rdx
  __int64 *v54; // rax
  __int64 **v55; // rsi
  __int64 *v56; // r13
  __int64 *v57; // r9
  __int64 v58; // rax
  int v59; // eax
  __int64 v60; // r14
  __int64 v61; // rbx
  _QWORD *v62; // rax
  double v63; // xmm4_8
  double v64; // xmm5_8
  unsigned int v65; // r15d
  __int64 v66; // rax
  int v67; // eax
  __int64 v68; // r14
  __int64 v69; // rdx
  __int64 v70; // rax
  __int64 *v71; // r9
  __int64 v72; // rax
  unsigned __int64 v73; // rcx
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rsi
  __int64 v77; // rsi
  __int64 v78; // rdx
  unsigned __int8 *v79; // rsi
  __int64 v80; // r8
  __int64 v81; // rax
  __int64 *v82; // [rsp+8h] [rbp-D8h]
  __int64 v83; // [rsp+10h] [rbp-D0h]
  unsigned int v84; // [rsp+10h] [rbp-D0h]
  unsigned int v85; // [rsp+10h] [rbp-D0h]
  __int64 v86; // [rsp+10h] [rbp-D0h]
  __int64 v87; // [rsp+18h] [rbp-C8h]
  __int64 v88; // [rsp+18h] [rbp-C8h]
  __int64 v89; // [rsp+18h] [rbp-C8h]
  __int64 *v90; // [rsp+18h] [rbp-C8h]
  __int64 *v91; // [rsp+20h] [rbp-C0h]
  __int64 v92; // [rsp+20h] [rbp-C0h]
  __int64 v93; // [rsp+20h] [rbp-C0h]
  __int64 v94; // [rsp+20h] [rbp-C0h]
  __int64 *v95; // [rsp+20h] [rbp-C0h]
  __int64 v96; // [rsp+20h] [rbp-C0h]
  unsigned __int64 *v97; // [rsp+20h] [rbp-C0h]
  __int64 *v98; // [rsp+20h] [rbp-C0h]
  __int64 v99; // [rsp+28h] [rbp-B8h]
  __int64 v100; // [rsp+28h] [rbp-B8h]
  __int64 v101; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v102; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v103; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v104; // [rsp+48h] [rbp-98h]
  __int64 *v105; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v106; // [rsp+58h] [rbp-88h]
  __int16 v107; // [rsp+60h] [rbp-80h]
  const char *v108; // [rsp+70h] [rbp-70h] BYREF
  __int64 v109; // [rsp+78h] [rbp-68h]
  unsigned __int64 v110; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v111; // [rsp+88h] [rbp-58h]
  const char **v112; // [rsp+90h] [rbp-50h] BYREF
  char *v113; // [rsp+98h] [rbp-48h]
  unsigned __int64 v114; // [rsp+A0h] [rbp-40h] BYREF
  unsigned int v115; // [rsp+A8h] [rbp-38h]

  v12 = a2;
  v13 = a4;
  v16 = *(_BYTE **)(a2 - 24);
  v17 = v16[16];
  if ( v17 == 13 )
  {
    v18 = (__int64)(v16 + 24);
  }
  else if ( *(_BYTE *)(*(_QWORD *)v16 + 8LL) != 16
         || v17 > 0x10u
         || (v66 = sub_15A1020(v16, a2, *(_QWORD *)v16, a4)) == 0
         || (v18 = v66 + 24, *(_BYTE *)(v66 + 16) != 13) )
  {
    v19 = *(unsigned __int16 *)(a2 + 18);
    BYTE1(v19) &= ~0x80u;
    goto LABEL_5;
  }
  v19 = *(unsigned __int16 *)(a2 + 18);
  BYTE1(v19) &= ~0x80u;
  if ( v19 == 40 )
  {
    v48 = *(_DWORD *)(v18 + 8);
    if ( v48 <= 0x40 )
      v49 = *(_QWORD *)v18 == 0;
    else
      v49 = v48 == (unsigned int)sub_16A57B0(v18);
  }
  else
  {
    if ( v19 != 38 )
    {
LABEL_5:
      if ( (unsigned int)(v19 - 32) > 1 )
        return 0;
      v20 = *(__int64 **)(a2 - 48);
      v21 = *(__int64 ***)a3;
      if ( *v20 != *(_QWORD *)a3 || *((_BYTE *)v21 + 8) != 11 )
        return 0;
      v83 = *(_QWORD *)(a2 - 48);
      v91 = *(__int64 **)(a2 - 24);
      sub_14C2530((__int64)&v108, v20, a1[333], 0, a1[330], a3, a1[332], 0);
      sub_14C2530((__int64)&v112, v91, a1[333], 0, a1[330], a3, a1[332], 0);
      v22 = v109;
      v23 = v83;
      if ( (unsigned int)v109 <= 0x40 )
      {
        if ( v108 != (const char *)v112 )
          goto LABEL_36;
      }
      else
      {
        v84 = v109;
        v87 = v23;
        v24 = sub_16A5220((__int64)&v108, (const void **)&v112);
        v23 = v87;
        v22 = v84;
        if ( !v24 )
          goto LABEL_36;
      }
      if ( v111 <= 0x40 )
      {
        if ( v110 == v114 )
        {
LABEL_12:
          v106 = v22;
          if ( v22 > 0x40 )
          {
            v86 = v23;
            sub_16A4FD0((__int64)&v105, (const void **)&v108);
            v22 = v106;
            v23 = v86;
            if ( v106 > 0x40 )
            {
              sub_16A89F0((__int64 *)&v105, (__int64 *)&v110);
              v22 = v106;
              v27 = (__int64)v105;
              v23 = v86;
              v102 = v106;
              v101 = (__int64)v105;
              if ( v106 > 0x40 )
              {
                sub_16A4FD0((__int64)&v105, (const void **)&v101);
                v22 = v106;
                v23 = v86;
                if ( v106 > 0x40 )
                {
                  sub_16A8F40((__int64 *)&v105);
                  v28 = (unsigned __int64)v105;
                  v23 = v86;
                  v104 = v106;
                  v103 = (unsigned __int64)v105;
                  if ( v106 > 0x40 )
                  {
                    v82 = v105;
                    v67 = sub_16A5940((__int64)&v103);
                    v30 = v86;
                    if ( v67 == 1 )
                    {
                      if ( !v13 )
                      {
LABEL_85:
                        if ( v103 )
                          j_j___libc_free_0_0(v103);
                        goto LABEL_87;
                      }
LABEL_18:
                      v31 = a1[1];
                      v107 = 257;
                      v32 = sub_172B670(v31, v30, (__int64)v91, (__int64 *)&v105, *(double *)a5.m128_u64, a6, a7);
                      if ( (int)sub_16A9900((__int64)&v110, &v103) >= 0 )
                      {
                        v33 = a1[1];
                        v107 = 257;
                        v92 = v33;
                        v34 = sub_15A1070((__int64)v21, (__int64)&v103);
                        v32 = sub_1729500(v92, v32, v34, (__int64 *)&v105, *(double *)a5.m128_u64, a6, a7);
                      }
                      v35 = a1[1];
                      v107 = 257;
                      v93 = v35;
                      if ( v104 > 0x40 )
                      {
                        v37 = sub_16A58A0((__int64)&v103);
                      }
                      else
                      {
                        _RDX = v103;
                        v37 = 64;
                        __asm { tzcnt   rcx, rdx }
                        if ( v103 )
                          v37 = _RCX;
                        if ( v37 > v104 )
                          v37 = v104;
                      }
                      v39 = sub_159C470((__int64)v21, v37, 0);
                      v40 = (__int64)sub_172C310(
                                       v93,
                                       (__int64)v32,
                                       v39,
                                       (__int64 *)&v105,
                                       0,
                                       *(double *)a5.m128_u64,
                                       a6,
                                       a7);
                      v41 = *(unsigned __int16 *)(v12 + 18);
                      BYTE1(v41) &= ~0x80u;
                      if ( v41 == 32 )
                      {
                        v80 = a1[1];
                        v107 = 257;
                        v100 = v80;
                        v81 = sub_159C470((__int64)v21, 1, 0);
                        v40 = (__int64)sub_172B670(v100, v40, v81, (__int64 *)&v105, *(double *)a5.m128_u64, a6, a7);
                      }
                      sub_164B7C0(v40, v12);
                      v12 = *(_QWORD *)(a3 + 8);
                      if ( v12 )
                      {
                        v42 = *a1;
                        do
                        {
                          v43 = sub_1648700(v12);
                          sub_170B990(v42, (__int64)v43);
                          v12 = *(_QWORD *)(v12 + 8);
                        }
                        while ( v12 );
                        if ( a3 == v40 )
                          v40 = sub_1599EF0(*(__int64 ***)a3);
                        v12 = a3;
                        sub_164D160(a3, v40, a5, a6, a7, a8, v44, v45, a11, a12);
                      }
                      v46 = v12;
                      if ( v104 <= 0x40 )
                        goto LABEL_88;
                      goto LABEL_85;
                    }
                    if ( v82 )
                      j_j___libc_free_0_0(v82);
                    goto LABEL_106;
                  }
LABEL_16:
                  v99 = v23;
                  v29 = sub_39FAC40(v28);
                  v30 = v99;
                  if ( v29 == 1 )
                  {
                    if ( !v13 )
                    {
LABEL_87:
                      v46 = v12;
LABEL_88:
                      if ( v102 > 0x40 && v101 )
                        j_j___libc_free_0_0(v101);
                      if ( v115 > 0x40 && v114 )
                        j_j___libc_free_0_0(v114);
                      if ( (unsigned int)v113 > 0x40 && v112 )
                        j_j___libc_free_0_0(v112);
                      if ( v111 > 0x40 && v110 )
                        j_j___libc_free_0_0(v110);
                      if ( (unsigned int)v109 > 0x40 && v108 )
                        j_j___libc_free_0_0(v108);
                      return v46;
                    }
                    goto LABEL_18;
                  }
LABEL_106:
                  if ( v102 > 0x40 && v101 )
                    j_j___libc_free_0_0(v101);
                  goto LABEL_36;
                }
                v27 = (__int64)v105;
              }
LABEL_15:
              v104 = v22;
              v28 = ~v27 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v22);
              v103 = v28;
              goto LABEL_16;
            }
            v26 = (unsigned __int64)v105;
          }
          else
          {
            v26 = (unsigned __int64)v108;
          }
          v27 = v110 | v26;
          v102 = v22;
          v101 = v27;
          goto LABEL_15;
        }
      }
      else
      {
        v85 = v22;
        v88 = v23;
        v25 = sub_16A5220((__int64)&v110, (const void **)&v114);
        v23 = v88;
        v22 = v85;
        if ( v25 )
          goto LABEL_12;
      }
LABEL_36:
      if ( v115 > 0x40 && v114 )
        j_j___libc_free_0_0(v114);
      if ( (unsigned int)v113 > 0x40 && v112 )
        j_j___libc_free_0_0(v112);
      if ( v111 > 0x40 && v110 )
        j_j___libc_free_0_0(v110);
      if ( (unsigned int)v109 > 0x40 && v108 )
        j_j___libc_free_0_0(v108);
      return 0;
    }
    v65 = *(_DWORD *)(v18 + 8);
    if ( v65 <= 0x40 )
      v49 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v65) == *(_QWORD *)v18;
    else
      v49 = v65 == (unsigned int)sub_16A58F0(v18);
  }
  if ( !v49 )
    return 0;
  if ( !v13 )
    return a2;
  v50 = *(__int64 **)(a2 - 48);
  v51 = *v50;
  v52 = sub_16431D0(*v50);
  v89 = sub_15A0680(v51, (unsigned int)(v52 - 1), 0);
  v94 = a1[1];
  v108 = sub_1649960((__int64)v50);
  v109 = v53;
  v112 = &v108;
  LOWORD(v114) = 773;
  v113 = ".lobit";
  v54 = sub_172C310(v94, (__int64)v50, v89, (__int64 *)&v112, 0, *(double *)a5.m128_u64, a6, a7);
  v55 = *(__int64 ***)a3;
  v56 = v54;
  if ( *v54 != *(_QWORD *)a3 )
  {
    v57 = (__int64 *)a1[1];
    LOWORD(v110) = 257;
    if ( v55 != (__int64 **)*v54 )
    {
      v95 = v57;
      if ( *((_BYTE *)v54 + 16) > 0x10u )
      {
        LOWORD(v114) = 257;
        v70 = sub_15FE0A0(v54, (__int64)v55, 0, (__int64)&v112, 0);
        v71 = v95;
        v56 = (__int64 *)v70;
        v72 = v95[1];
        if ( v72 )
        {
          v90 = v95;
          v97 = (unsigned __int64 *)v95[2];
          sub_157E9D0(v72 + 40, (__int64)v56);
          v71 = v90;
          v73 = *v97;
          v74 = v56[3] & 7;
          v56[4] = (__int64)v97;
          v73 &= 0xFFFFFFFFFFFFFFF8LL;
          v56[3] = v73 | v74;
          *(_QWORD *)(v73 + 8) = v56 + 3;
          *v97 = *v97 & 7 | (unsigned __int64)(v56 + 3);
        }
        v98 = v71;
        sub_164B780((__int64)v56, (__int64 *)&v108);
        v105 = v56;
        if ( !v98[10] )
          sub_4263D6(v56, &v108, v75);
        ((void (__fastcall *)(__int64 *, __int64 **))v98[11])(v98 + 8, &v105);
        v76 = *v98;
        if ( *v98 )
        {
          v105 = (__int64 *)*v98;
          sub_1623A60((__int64)&v105, v76, 2);
          v77 = v56[6];
          v78 = (__int64)(v56 + 6);
          if ( v77 )
          {
            sub_161E7C0((__int64)(v56 + 6), v77);
            v78 = (__int64)(v56 + 6);
          }
          v79 = (unsigned __int8 *)v105;
          v56[6] = (__int64)v105;
          if ( v79 )
            sub_1623210((__int64)&v105, v79, v78);
        }
      }
      else
      {
        v56 = (__int64 *)sub_15A4750((__int64 ***)v54, v55, 0);
        v58 = sub_14DBA30((__int64)v56, v95[12], 0);
        if ( v58 )
          v56 = (__int64 *)v58;
      }
    }
  }
  v59 = *(unsigned __int16 *)(v12 + 18);
  BYTE1(v59) &= ~0x80u;
  if ( v59 == 38 )
  {
    v68 = sub_15A0680(*v56, 1, 0);
    v96 = a1[1];
    v108 = sub_1649960((__int64)v56);
    v109 = v69;
    LOWORD(v114) = 773;
    v112 = &v108;
    v113 = ".not";
    v56 = (__int64 *)sub_172B670(v96, (__int64)v56, v68, (__int64 *)&v112, *(double *)a5.m128_u64, a6, a7);
  }
  v60 = *(_QWORD *)(a3 + 8);
  v46 = a3;
  if ( !v60 )
    return 0;
  v61 = *a1;
  do
  {
    v62 = sub_1648700(v60);
    sub_170B990(v61, (__int64)v62);
    v60 = *(_QWORD *)(v60 + 8);
  }
  while ( v60 );
  if ( (__int64 *)a3 == v56 )
    v56 = (__int64 *)sub_1599EF0(*(__int64 ***)a3);
  sub_164D160(a3, (__int64)v56, a5, a6, a7, a8, v63, v64, a11, a12);
  return v46;
}
