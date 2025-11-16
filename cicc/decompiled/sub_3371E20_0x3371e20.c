// Function: sub_3371E20
// Address: 0x3371e20
//
void __fastcall sub_3371E20(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        _BYTE *a8,
        int a9)
{
  unsigned __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // r13
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  _BYTE *v14; // rax
  _BYTE *v15; // rdx
  __int64 v16; // rbx
  _BYTE *i; // rdx
  unsigned int v18; // ebx
  unsigned int k; // r13d
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 (__fastcall *v23)(__int64, __int64, __int64, __int64, unsigned __int64); // r9
  unsigned __int16 v24; // r9
  __int64 v25; // r8
  __int64 v26; // rax
  unsigned int v27; // r12d
  unsigned __int16 *v28; // rax
  __int64 (__fastcall *v29)(__int64, __int64, unsigned int); // rax
  __int64 (*v30)(); // rax
  __int64 v31; // rdx
  __int64 v32; // rdi
  char v33; // al
  __int64 (__fastcall *v34)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 (__fastcall *v38)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v39; // rax
  __int64 v40; // rcx
  __int64 (__fastcall *v41)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v42; // rax
  __int64 v43; // rcx
  __int64 (__fastcall *v44)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v45; // rax
  unsigned __int64 v46; // rcx
  _QWORD *v47; // rbx
  _QWORD *j; // rax
  __int64 v49; // r12
  __int64 v50; // r11
  unsigned int v51; // r15d
  int v52; // eax
  int v53; // edx
  __int64 v54; // rbx
  __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // r10
  __int64 v58; // rdx
  __int64 *v59; // rax
  unsigned int *v60; // rdx
  __int64 v61; // r10
  __int64 v62; // r14
  __int64 v63; // rbx
  __int64 v64; // rbx
  __int64 v65; // rax
  __int128 v66; // rax
  __int64 v67; // rax
  _QWORD *v68; // rdi
  _QWORD *v69; // rax
  __int64 v70; // rax
  int v71; // edx
  __int64 v72; // rdx
  __int64 v73; // rdx
  __int64 v74; // rdx
  unsigned __int64 v75; // rdx
  __int128 v76; // [rsp-10h] [rbp-2C0h]
  __int128 v77; // [rsp-10h] [rbp-2C0h]
  __int64 v78; // [rsp+0h] [rbp-2B0h]
  __int64 v79; // [rsp+8h] [rbp-2A8h]
  __int16 v80; // [rsp+12h] [rbp-29Eh]
  __int64 v81; // [rsp+18h] [rbp-298h]
  unsigned __int64 v82; // [rsp+28h] [rbp-288h]
  __int64 v83; // [rsp+38h] [rbp-278h]
  __int64 v84; // [rsp+40h] [rbp-270h]
  __int64 v85; // [rsp+50h] [rbp-260h]
  unsigned __int64 v86; // [rsp+50h] [rbp-260h]
  unsigned __int64 v87; // [rsp+50h] [rbp-260h]
  unsigned __int64 v88; // [rsp+50h] [rbp-260h]
  __int64 v89; // [rsp+50h] [rbp-260h]
  unsigned __int64 v91; // [rsp+68h] [rbp-248h]
  int v92; // [rsp+68h] [rbp-248h]
  __int64 v93; // [rsp+70h] [rbp-240h]
  int v94; // [rsp+70h] [rbp-240h]
  __int64 v95; // [rsp+78h] [rbp-238h]
  unsigned __int16 v96; // [rsp+78h] [rbp-238h]
  int v97; // [rsp+78h] [rbp-238h]
  unsigned __int16 v98; // [rsp+78h] [rbp-238h]
  int v99; // [rsp+84h] [rbp-22Ch]
  unsigned int v100; // [rsp+84h] [rbp-22Ch]
  int v101; // [rsp+88h] [rbp-228h]
  __int64 v102; // [rsp+90h] [rbp-220h]
  __int64 v103; // [rsp+90h] [rbp-220h]
  __int128 v104; // [rsp+90h] [rbp-220h]
  __int64 v105; // [rsp+A0h] [rbp-210h]
  __int64 v106; // [rsp+A0h] [rbp-210h]
  __int128 v107; // [rsp+A0h] [rbp-210h]
  unsigned int v108; // [rsp+B0h] [rbp-200h]
  int v109; // [rsp+B0h] [rbp-200h]
  int v110; // [rsp+B4h] [rbp-1FCh]
  unsigned __int16 v112; // [rsp+EAh] [rbp-1C6h] BYREF
  unsigned int v113; // [rsp+ECh] [rbp-1C4h] BYREF
  __int64 v114; // [rsp+F0h] [rbp-1C0h] BYREF
  __int64 v115; // [rsp+F8h] [rbp-1B8h]
  __int64 v116; // [rsp+100h] [rbp-1B0h] BYREF
  __int64 v117; // [rsp+108h] [rbp-1A8h]
  __int64 v118; // [rsp+110h] [rbp-1A0h] BYREF
  __int64 v119; // [rsp+118h] [rbp-198h]
  __int64 v120; // [rsp+120h] [rbp-190h] BYREF
  __int64 v121; // [rsp+128h] [rbp-188h]
  __int64 v122; // [rsp+130h] [rbp-180h]
  __int64 v123; // [rsp+138h] [rbp-178h]
  __int64 v124; // [rsp+140h] [rbp-170h]
  int v125; // [rsp+148h] [rbp-168h]
  __int64 v126; // [rsp+150h] [rbp-160h]
  int v127; // [rsp+158h] [rbp-158h]
  _BYTE *v128; // [rsp+160h] [rbp-150h] BYREF
  __int64 v129; // [rsp+168h] [rbp-148h]
  _BYTE v130[128]; // [rsp+170h] [rbp-140h] BYREF
  _QWORD *v131; // [rsp+1F0h] [rbp-C0h] BYREF
  __int64 v132; // [rsp+1F8h] [rbp-B8h]
  _QWORD v133[22]; // [rsp+200h] [rbp-B0h] BYREF

  v10 = a4;
  v11 = a1;
  v12 = *(_QWORD *)(a4 + 16);
  v13 = *(unsigned int *)(a1 + 120);
  v82 = a3;
  v102 = v12;
  v108 = a3;
  v110 = *(_DWORD *)(a1 + 120);
  v91 = v13;
  v128 = v130;
  v129 = 0x800000000LL;
  if ( v13 )
  {
    v14 = v130;
    v15 = v130;
    if ( v13 > 8 )
    {
      sub_C8D5F0((__int64)&v128, v130, v13, 0x10u, a5, a6);
      v15 = v128;
      v14 = &v128[16 * (unsigned int)v129];
    }
    v16 = 2 * v91;
    for ( i = &v15[16 * v91]; i != v14; v14 += 16 )
    {
      if ( v14 )
      {
        *(_QWORD *)v14 = 0;
        *((_DWORD *)v14 + 2) = 0;
      }
    }
    LODWORD(v129) = v110;
    v99 = *(_DWORD *)(a1 + 8);
    if ( !v99 )
    {
      v131 = v133;
      v132 = 0x800000000LL;
LABEL_42:
      if ( v91 > 8 )
        sub_C8D5F0((__int64)&v131, v133, v91, 0x10u, a5, a6);
      v47 = &v131[v16];
      for ( j = &v131[2 * (unsigned int)v132]; v47 != j; j += 2 )
      {
        if ( j )
        {
          *j = 0;
          *((_DWORD *)j + 2) = 0;
        }
      }
      v49 = v10;
      v50 = a7;
      v89 = v11;
      v51 = 0;
      WORD1(v11) = v80;
      LODWORD(v132) = v110;
      do
      {
        v60 = (unsigned int *)(*(_QWORD *)(v89 + 112) + 4LL * v51);
        v61 = 16LL * v51;
        v62 = *(_QWORD *)&v128[v61];
        v63 = 16LL * *(unsigned int *)&v128[v61 + 8];
        if ( v50 )
        {
          a7 = v50;
          v100 = *v60;
          v109 = *(_DWORD *)(v50 + 8);
          v101 = *(_DWORD *)(a6 + 8);
          v103 = *(_QWORD *)a6;
          v92 = *(_DWORD *)&v128[v61 + 8];
          v106 = *(_QWORD *)v50;
          v52 = sub_33E5110(v49, 1, 0, 262, 0);
          v94 = v53;
          v120 = v103;
          v97 = v52;
          LODWORD(v121) = v101;
          v54 = *(_QWORD *)(v62 + 48) + v63;
          LOWORD(v11) = *(_WORD *)v54;
          v122 = sub_33F0B60(v49, v100, (unsigned int)v11, *(_QWORD *)(v54 + 8));
          v123 = v55;
          v126 = v106;
          *((_QWORD *)&v76 + 1) = 3 - ((v106 == 0) - 1LL);
          *(_QWORD *)&v76 = &v120;
          v127 = v109;
          v124 = v62;
          v125 = v92;
          v56 = sub_3411630(v49, 49, a5, v97, v94, v92, v76);
          v50 = a7;
          v57 = 2LL * v51;
          v58 = v56;
          *(_QWORD *)a7 = v56;
          *(_DWORD *)(a7 + 8) = 1;
        }
        else
        {
          v64 = *(_QWORD *)(v62 + 48) + v63;
          a7 = 0;
          v65 = v84;
          LOWORD(v65) = *(_WORD *)v64;
          v107 = (__int128)_mm_loadu_si128((const __m128i *)&v128[v61]);
          v84 = v65;
          v104 = *(_OWORD *)a6;
          *(_QWORD *)&v66 = sub_33F0B60(v49, *v60, (unsigned int)v65, *(_QWORD *)(v64 + 8));
          v67 = sub_340F900(v49, 49, a5, 1, 0, DWORD2(v104), v104, v66, v107);
          v57 = 2LL * v51;
          v50 = 0;
          v58 = v67;
        }
        ++v51;
        v59 = &v131[v57];
        *v59 = v58;
        *((_DWORD *)v59 + 2) = 0;
      }
      while ( v110 != v51 );
      v68 = v131;
      LODWORD(v10) = v49;
      goto LABEL_56;
    }
  }
  else
  {
    v99 = *(_DWORD *)(a1 + 8);
    if ( !v99 )
    {
      v132 = 0x800000000LL;
      v131 = v133;
      v68 = v133;
      goto LABEL_56;
    }
  }
  v105 = v10;
  v18 = 0;
  for ( k = 0; k != v99; ++k )
  {
    v27 = *(_DWORD *)(*(_QWORD *)(a1 + 144) + 4LL * k);
    v28 = (unsigned __int16 *)(*(_QWORD *)(a1 + 80) + 2LL * k);
    if ( *(_BYTE *)(a1 + 180) )
    {
      v21 = *v28;
      v22 = *(_QWORD *)v102;
      v23 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, unsigned __int64))(*(_QWORD *)v102 + 744LL);
      v95 = *(_QWORD *)(v105 + 64);
      if ( v23 != sub_2FE9BB0 )
      {
        v24 = v23(v102, v95, *(unsigned int *)(a1 + 176), (unsigned __int16)v21, 0);
        goto LABEL_13;
      }
      LOWORD(v114) = v21;
      v115 = 0;
      if ( (_WORD)v21 )
      {
        v24 = *(_WORD *)(v102 + 2 * v21 + 2852);
        goto LABEL_13;
      }
      v85 = v22;
      if ( sub_30070B0((__int64)&v114) )
      {
        v132 = 0;
        LOWORD(v118) = 0;
        LOWORD(v131) = 0;
        sub_2FE8D10(
          v102,
          v95,
          (unsigned int)v114,
          0,
          (__int64 *)&v131,
          (unsigned int *)&v120,
          (unsigned __int16 *)&v118);
      }
      else
      {
        if ( !sub_3007070((__int64)&v114) )
          goto LABEL_81;
        v34 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(v85 + 592);
        if ( v34 == sub_2D56A50 )
        {
          sub_2FE6CC0((__int64)&v131, v102, v95, v114, v115);
          v35 = v83;
          LOWORD(v35) = v132;
          v36 = v133[0];
          v83 = v35;
        }
        else
        {
          v83 = v34(v102, v95, v114, 0);
          v36 = v72;
        }
        v117 = v36;
        v37 = (unsigned __int16)v83;
        v116 = v83;
        if ( (_WORD)v83 )
          goto LABEL_54;
        v86 = v36;
        if ( !sub_30070B0((__int64)&v116) )
        {
          if ( !sub_3007070((__int64)&v116) )
            goto LABEL_81;
          v38 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v102 + 592LL);
          if ( v38 == sub_2D56A50 )
          {
            sub_2FE6CC0((__int64)&v131, v102, v95, v116, v117);
            v39 = v81;
            LOWORD(v39) = v132;
            v40 = v133[0];
            v81 = v39;
          }
          else
          {
            v81 = v38(v102, v95, v116, v86);
            v40 = v73;
          }
          v119 = v40;
          v37 = (unsigned __int16)v81;
          v118 = v81;
          if ( !(_WORD)v81 )
          {
            v87 = v40;
            if ( sub_30070B0((__int64)&v118) )
            {
              LOWORD(v131) = 0;
              LOWORD(v113) = 0;
              v132 = 0;
              sub_2FE8D10(
                v102,
                v95,
                (unsigned int)v118,
                v87,
                (__int64 *)&v131,
                (unsigned int *)&v120,
                (unsigned __int16 *)&v113);
              v24 = v113;
              goto LABEL_13;
            }
            if ( !sub_3007070((__int64)&v118) )
              goto LABEL_81;
            v41 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v102 + 592LL);
            if ( v41 == sub_2D56A50 )
            {
              sub_2FE6CC0((__int64)&v131, v102, v95, v118, v119);
              v42 = v79;
              LOWORD(v42) = v132;
              v43 = v133[0];
              v79 = v42;
            }
            else
            {
              v79 = v41(v102, v95, v118, v87);
              v43 = v74;
            }
            v121 = v43;
            v37 = (unsigned __int16)v79;
            v120 = v79;
            if ( !(_WORD)v79 )
            {
              v88 = v43;
              if ( sub_30070B0((__int64)&v120) )
              {
                LOWORD(v131) = 0;
                v112 = 0;
                v132 = 0;
                sub_2FE8D10(v102, v95, (unsigned int)v120, v88, (__int64 *)&v131, &v113, &v112);
                v24 = v112;
              }
              else
              {
                if ( !sub_3007070((__int64)&v120) )
LABEL_81:
                  BUG();
                v44 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v102 + 592LL);
                if ( v44 == sub_2D56A50 )
                {
                  sub_2FE6CC0((__int64)&v131, v102, v95, v120, v121);
                  v45 = v78;
                  LOWORD(v45) = v132;
                  v46 = v133[0];
                  v78 = v45;
                }
                else
                {
                  v78 = v44(v102, v95, v120, v88);
                  v46 = v75;
                }
                v24 = sub_2FE98B0(v102, v95, (unsigned int)v78, v46);
              }
LABEL_13:
              if ( a9 != 215 )
                goto LABEL_14;
              goto LABEL_17;
            }
          }
LABEL_54:
          v24 = *(_WORD *)(v102 + 2 * v37 + 2852);
          goto LABEL_13;
        }
        LOWORD(v118) = 0;
        LOWORD(v131) = 0;
        v132 = 0;
        sub_2FE8D10(
          v102,
          v95,
          (unsigned int)v116,
          v86,
          (__int64 *)&v131,
          (unsigned int *)&v120,
          (unsigned __int16 *)&v118);
      }
      v24 = v118;
      goto LABEL_13;
    }
    v24 = *v28;
    if ( a9 != 215 )
      goto LABEL_14;
LABEL_17:
    v29 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v102 + 1448LL);
    if ( v29 == sub_2FE39F0 )
    {
      v30 = *(__int64 (**)())(*(_QWORD *)v102 + 1432LL);
      v31 = *(_QWORD *)(a2 + 48) + 16LL * v108;
      v32 = v93;
      LOWORD(v32) = *(_WORD *)v31;
      v93 = v32;
      if ( v30 == sub_2FE34A0 )
        goto LABEL_14;
      v96 = v24;
      v33 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD))v30)(
              v102,
              (unsigned int)v32,
              *(_QWORD *)(v31 + 8),
              v24,
              0);
      v24 = v96;
    }
    else
    {
      v98 = v24;
      v82 = v108 | v82 & 0xFFFFFFFF00000000LL;
      v33 = ((__int64 (__fastcall *)(__int64, __int64, unsigned __int64, _QWORD, _QWORD))v29)(v102, a2, v82, v24, 0);
      v24 = v98;
    }
    if ( v33 )
      a9 = 214;
LABEL_14:
    v25 = v18;
    v18 += v27;
    v26 = k + v108;
    v9 = v26 | v9 & 0xFFFFFFFF00000000LL;
    sub_33695F0(v105, a5, a2, v9, (__m128i *)&v128[16 * v25], v27, v24, a8, *(_QWORD *)(a1 + 176), a9);
  }
  v11 = a1;
  v10 = v105;
  v131 = v133;
  v132 = 0x800000000LL;
  if ( v91 )
  {
    v16 = 2 * v91;
    goto LABEL_42;
  }
  v68 = v133;
LABEL_56:
  if ( v110 == 1 || a7 )
  {
    v69 = &v68[2 * (unsigned int)(v110 - 1)];
    *(_QWORD *)a6 = *v69;
    *(_DWORD *)(a6 + 8) = *((_DWORD *)v69 + 2);
  }
  else
  {
    *((_QWORD *)&v77 + 1) = (unsigned int)v132;
    *(_QWORD *)&v77 = v68;
    v70 = sub_33FC220(v10, 2, a5, 1, 0, a6, v77);
    v68 = v131;
    *(_QWORD *)a6 = v70;
    *(_DWORD *)(a6 + 8) = v71;
  }
  if ( v68 != v133 )
    _libc_free((unsigned __int64)v68);
  if ( v128 != v130 )
    _libc_free((unsigned __int64)v128);
}
