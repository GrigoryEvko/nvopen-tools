// Function: sub_1940CE0
// Address: 0x1940ce0
//
void __fastcall sub_1940CE0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r14
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // r13
  int v16; // eax
  unsigned __int64 *v17; // rdi
  unsigned __int64 v18; // rax
  bool v19; // zf
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 v22; // r13
  __int64 v23; // r12
  __int64 v24; // rdx
  _BOOL4 v25; // ebx
  bool v26; // r11
  __int64 v27; // rdx
  _BOOL8 v28; // r14
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rdi
  __int64 v35; // r8
  __int64 v36; // rdi
  _QWORD *v37; // rax
  __int64 v38; // rax
  int v39; // eax
  _QWORD *v40; // rbx
  _QWORD *v41; // r12
  __int64 v42; // rax
  int v43; // r10d
  unsigned int v44; // eax
  unsigned int v45; // edx
  _QWORD *v46; // rax
  const char *v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int16 v50; // r10
  __int64 v51; // r14
  __int64 v52; // rax
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  const char *v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // r14
  __int64 v60; // rax
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // r9
  const char *v64; // rax
  __int64 v65; // rdx
  _QWORD *v66; // r14
  _QWORD **v67; // rax
  __int64 *v68; // rax
  __int64 v69; // rax
  __int16 v70; // r10
  double v71; // xmm4_8
  double v72; // xmm5_8
  __int64 v73; // rax
  double v74; // xmm4_8
  double v75; // xmm5_8
  __int64 v76; // rbx
  __int64 v77; // rax
  __int64 v78; // r14
  _QWORD *v79; // rax
  double v80; // xmm4_8
  double v81; // xmm5_8
  __int64 v82; // r9
  __int64 v83; // rsi
  unsigned int v84; // eax
  int v85; // edi
  unsigned int v86; // r8d
  unsigned int v87; // eax
  __int16 v88; // [rsp+Ch] [rbp-1C4h]
  __int16 v89; // [rsp+10h] [rbp-1C0h]
  _QWORD *v90; // [rsp+10h] [rbp-1C0h]
  __int16 v91; // [rsp+18h] [rbp-1B8h]
  __int64 v92; // [rsp+28h] [rbp-1A8h]
  __int64 v93; // [rsp+28h] [rbp-1A8h]
  __int64 v94; // [rsp+30h] [rbp-1A0h]
  _BOOL8 v95; // [rsp+38h] [rbp-198h]
  __int64 v96; // [rsp+38h] [rbp-198h]
  _QWORD *v97; // [rsp+40h] [rbp-190h]
  __int64 v98; // [rsp+48h] [rbp-188h]
  __int64 v99; // [rsp+48h] [rbp-188h]
  _QWORD *v100; // [rsp+48h] [rbp-188h]
  __int64 v102; // [rsp+60h] [rbp-170h]
  __int64 v104; // [rsp+78h] [rbp-158h] BYREF
  __int64 v105; // [rsp+80h] [rbp-150h] BYREF
  __int64 v106; // [rsp+88h] [rbp-148h] BYREF
  __int64 v107; // [rsp+90h] [rbp-140h] BYREF
  __int64 v108; // [rsp+98h] [rbp-138h]
  __int64 v109; // [rsp+A0h] [rbp-130h]
  __int64 v110; // [rsp+B0h] [rbp-120h] BYREF
  char *v111; // [rsp+B8h] [rbp-118h]
  unsigned __int64 v112; // [rsp+C0h] [rbp-110h]
  _BYTE *v113; // [rsp+D0h] [rbp-100h] BYREF
  __int64 v114; // [rsp+D8h] [rbp-F8h]
  _BYTE v115[240]; // [rsp+E0h] [rbp-F0h] BYREF

  v10 = a2;
  v11 = **(_QWORD **)(a2 + 32);
  v113 = v115;
  v114 = 0x800000000LL;
  v12 = sub_157F280(v11);
  v14 = v13;
  v15 = v12;
  while ( v14 != v15 )
  {
    v112 = v15;
    v110 = 6;
    v111 = 0;
    if ( v15 != 0 && v15 != -8 && v15 != -16 )
      sub_164C220((__int64)&v110);
    v16 = v114;
    if ( (unsigned int)v114 >= HIDWORD(v114) )
    {
      sub_170B450((__int64)&v113, 0);
      v16 = v114;
    }
    v17 = (unsigned __int64 *)&v113[24 * v16];
    if ( v17 )
    {
      *v17 = 6;
      v17[1] = 0;
      v18 = v112;
      v19 = v112 == -8;
      v17[2] = v112;
      if ( v18 != 0 && !v19 && v18 != -16 )
        sub_1649AC0(v17, v110 & 0xFFFFFFFFFFFFFFF8LL);
      v16 = v114;
    }
    LODWORD(v114) = v16 + 1;
    if ( v112 != 0 && v112 != -8 && v112 != -16 )
      sub_1649B30(&v110);
    if ( !v15 )
      BUG();
    v20 = *(_QWORD *)(v15 + 32);
    if ( !v20 )
      BUG();
    v15 = 0;
    if ( *(_BYTE *)(v20 - 8) == 77 )
      v15 = v20 - 24;
  }
  if ( (_DWORD)v114 )
  {
    v21 = 0;
    v22 = 24LL * (unsigned int)v114;
    v102 = a2 + 56;
    do
    {
      v23 = *(_QWORD *)&v113[v21 + 16];
      if ( v23 && *(_BYTE *)(v23 + 16) == 77 )
      {
        v24 = (*(_BYTE *)(v23 + 23) & 0x40) != 0
            ? *(_QWORD *)(v23 - 8)
            : v23 - 24LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF);
        v25 = sub_1377F70(v102, *(_QWORD *)(v24 + 24LL * *(unsigned int *)(v23 + 56) + 8));
        v26 = v25;
        v27 = (*(_BYTE *)(v23 + 23) & 0x40) != 0
            ? *(_QWORD *)(v23 - 8)
            : v23 - 24LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF);
        v28 = v26;
        v29 = *(_QWORD *)(v27 + 24LL * v26);
        if ( *(_BYTE *)(v29 + 16) == 14 )
        {
          if ( (unsigned __int8)sub_193DF40(v29 + 24, &v104) )
          {
            v30 = (*(_BYTE *)(v23 + 23) & 0x40) != 0
                ? *(_QWORD *)(v23 - 8)
                : v23 - 24LL * (*(_DWORD *)(v23 + 20) & 0xFFFFFFF);
            v95 = !v25;
            v31 = *(_QWORD *)(v30 + 24 * v95);
            if ( *(_BYTE *)(v31 + 16) == 36 )
            {
              v32 = *(_QWORD *)(v31 - 24);
              if ( *(_BYTE *)(v32 + 16) == 14 )
              {
                v33 = *(_QWORD *)(v31 - 48);
                if ( v23 == v33 )
                {
                  if ( v33 )
                  {
                    if ( (unsigned __int8)sub_193DF40(v32 + 24, &v105) )
                    {
                      v34 = *(_QWORD *)(v31 + 8);
                      v35 = *(_QWORD *)(v34 + 8);
                      if ( v35 )
                      {
                        v98 = *(_QWORD *)(v34 + 8);
                        if ( !*(_QWORD *)(v35 + 8) )
                        {
                          v97 = sub_1648700(v34);
                          if ( *((_BYTE *)v97 + 16) == 76 || (v97 = sub_1648700(v98), *((_BYTE *)v97 + 16) == 76) )
                          {
                            v36 = v97[1];
                            if ( v36 )
                            {
                              if ( !*(_QWORD *)(v36 + 8) )
                              {
                                v37 = sub_1648700(v36);
                                v94 = (__int64)v37;
                                if ( *((_BYTE *)v37 + 16) == 26
                                  && sub_1377F70(v102, v37[5])
                                  && (!sub_1377F70(v102, *(_QWORD *)(v94 - 24))
                                   || !sub_1377F70(v102, *(_QWORD *)(v94 - 48))) )
                                {
                                  v38 = *(v97 - 3);
                                  if ( *(_BYTE *)(v38 + 16) == 14 )
                                  {
                                    if ( (unsigned __int8)sub_193DF40(v38 + 24, &v106) )
                                    {
                                      v39 = *((unsigned __int16 *)v97 + 9);
                                      BYTE1(v39) &= ~0x80u;
                                      switch ( v39 )
                                      {
                                        case 1:
                                        case 9:
                                          v43 = 32;
                                          goto LABEL_68;
                                        case 2:
                                        case 10:
                                          v43 = 38;
                                          goto LABEL_68;
                                        case 3:
                                        case 11:
                                          v43 = 39;
                                          goto LABEL_68;
                                        case 4:
                                        case 12:
                                          v43 = 40;
                                          goto LABEL_68;
                                        case 5:
                                        case 13:
                                          v43 = 41;
                                          goto LABEL_68;
                                        case 6:
                                        case 14:
                                          v43 = 33;
LABEL_68:
                                          if ( v104 != (int)v104 || v105 != (int)v105 || (int)v106 != v106 || !v105 )
                                            break;
                                          if ( v105 <= 0 )
                                          {
                                            if ( v104 <= v106 )
                                              break;
                                            v84 = v104 - v106;
                                            if ( (unsigned int)(v43 - 39) <= 1 )
                                            {
                                              v87 = v84 + 1;
                                              if ( !v87 )
                                                break;
                                              v85 = v105;
                                              v86 = v87 % -(int)v105;
                                            }
                                            else
                                            {
                                              v85 = v105;
                                              v45 = v84 % -(int)v105;
                                              v86 = v45;
                                              if ( (unsigned int)(v43 - 32) <= 1 )
                                                goto LABEL_99;
                                            }
                                            if ( v86 && v106 < v85 + (int)v106 )
                                              break;
                                          }
                                          else
                                          {
                                            if ( v104 >= v106 )
                                              break;
                                            v44 = v106 - v104;
                                            if ( (v43 == 41 || v43 == 38) && !++v44 )
                                              break;
                                            v45 = v44 % (unsigned int)v105;
                                            if ( (unsigned int)(v43 - 32) <= 1 )
                                            {
LABEL_99:
                                              if ( v45 )
                                                break;
                                              goto LABEL_79;
                                            }
                                            if ( v45 && v106 > (int)v105 + (int)v106 )
                                              break;
                                          }
LABEL_79:
                                          v91 = v43;
                                          v46 = (_QWORD *)sub_16498A0(v23);
                                          v92 = sub_1643350(v46);
                                          v47 = sub_1649960(v23);
                                          LOWORD(v112) = 773;
                                          v107 = (__int64)v47;
                                          v110 = (__int64)&v107;
                                          v108 = v48;
                                          v111 = ".int";
                                          v49 = sub_1648B60(64);
                                          v50 = v91;
                                          v99 = v49;
                                          if ( v49 )
                                          {
                                            sub_15F1EA0(v49, v92, 53, 0, 0, v23);
                                            *(_DWORD *)(v99 + 56) = 2;
                                            sub_164B780(v99, &v110);
                                            sub_1648880(v99, *(_DWORD *)(v99 + 56), 1);
                                            v50 = v91;
                                          }
                                          v89 = v50;
                                          v51 = *(_QWORD *)(sub_193FF80(v23) + 8 * v28);
                                          v52 = sub_159C470(v92, v104, 0);
                                          sub_1704F80(v99, v52, v51, v53, v54, v55);
                                          v56 = sub_1649960(v31);
                                          LOWORD(v112) = 773;
                                          v107 = (__int64)v56;
                                          v111 = ".int";
                                          v108 = v57;
                                          v110 = (__int64)&v107;
                                          v58 = sub_159C470(v92, v105, 0);
                                          v59 = sub_15FB440(11, (__int64 *)v99, v58, (__int64)&v110, v31);
                                          v60 = sub_193FF80(v23);
                                          v61 = v95;
                                          v96 = v59;
                                          sub_1704F80(v99, v59, *(_QWORD *)(v60 + 8 * v61), v61, v62, v63);
                                          v93 = sub_159C470(v92, v106, 0);
                                          v64 = sub_1649960((__int64)v97);
                                          LOWORD(v112) = 261;
                                          v107 = (__int64)v64;
                                          v108 = v65;
                                          v110 = (__int64)&v107;
                                          v66 = sub_1648A60(56, 2u);
                                          if ( v66 )
                                          {
                                            v67 = *(_QWORD ***)v96;
                                            if ( *(_BYTE *)(*(_QWORD *)v96 + 8LL) == 16 )
                                            {
                                              v88 = v89;
                                              v90 = v67[4];
                                              v68 = (__int64 *)sub_1643320(*v67);
                                              v69 = (__int64)sub_16463B0(v68, (unsigned int)v90);
                                              v70 = v88;
                                            }
                                            else
                                            {
                                              v69 = sub_1643320(*v67);
                                              v70 = v89;
                                            }
                                            sub_15FEC10((__int64)v66, v69, 51, v70, v96, v93, (__int64)&v110, v94);
                                          }
                                          v107 = 6;
                                          v108 = 0;
                                          v109 = v23;
                                          if ( v23 != -16 && v23 != -8 )
                                            sub_164C220((__int64)&v107);
                                          sub_164B7C0((__int64)v66, (__int64)v97);
                                          sub_164D160((__int64)v97, (__int64)v66, a3, a4, a5, a6, v71, v72, a9, a10);
                                          sub_1AEB370(v97, *(_QWORD *)(a1 + 32));
                                          v73 = sub_1599EF0(*(__int64 ***)v31);
                                          sub_164D160(v31, v73, a3, a4, a5, a6, v74, v75, a9, a10);
                                          sub_1AEB370(v31, *(_QWORD *)(a1 + 32));
                                          if ( v109 )
                                          {
                                            v76 = *(_QWORD *)v23;
                                            v110 = (__int64)"indvar.conv";
                                            LOWORD(v112) = 259;
                                            v77 = sub_157EE30(*(_QWORD *)(v23 + 40));
                                            v78 = v77;
                                            if ( v77 )
                                              v78 = v77 - 24;
                                            v79 = sub_1648A60(56, 1u);
                                            v82 = (__int64)v79;
                                            if ( v79 )
                                            {
                                              v83 = v99;
                                              v100 = v79;
                                              sub_15FCE10((__int64)v79, v83, v76, (__int64)&v110, v78);
                                              v82 = (__int64)v100;
                                            }
                                            sub_164D160(v23, v82, a3, a4, a5, a6, v80, v81, a9, a10);
                                            sub_1AEB370(v23, *(_QWORD *)(a1 + 32));
                                          }
                                          *(_BYTE *)(a1 + 448) = 1;
                                          if ( v109 != 0 && v109 != -8 && v109 != -16 )
                                            sub_1649B30(&v107);
                                          break;
                                        default:
                                          break;
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      v21 += 24;
    }
    while ( v21 != v22 );
    v10 = a2;
    if ( !*(_BYTE *)(a1 + 448) )
      goto LABEL_51;
  }
  else if ( !*(_BYTE *)(a1 + 448) )
  {
    goto LABEL_56;
  }
  sub_1465150(*(_QWORD *)(a1 + 8), v10);
LABEL_51:
  v40 = v113;
  v41 = &v113[24 * (unsigned int)v114];
  if ( v113 == (_BYTE *)v41 )
    goto LABEL_57;
  do
  {
    v42 = *(v41 - 1);
    v41 -= 3;
    if ( v42 != -8 && v42 != 0 && v42 != -16 )
      sub_1649B30(v41);
  }
  while ( v40 != v41 );
LABEL_56:
  v41 = v113;
LABEL_57:
  if ( v41 != (_QWORD *)v115 )
    _libc_free((unsigned __int64)v41);
}
