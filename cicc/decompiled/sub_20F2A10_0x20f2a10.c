// Function: sub_20F2A10
// Address: 0x20f2a10
//
__int64 __fastcall sub_20F2A10(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rdx
  __int64 result; // rax
  __int64 v8; // r15
  int v11; // ecx
  __int64 (*v12)(void); // rax
  unsigned int v13; // eax
  __int64 *v14; // rax
  __int64 *v15; // r8
  int v16; // r11d
  __int64 *v17; // r12
  unsigned __int8 v18; // dl
  unsigned int v19; // ebx
  __int64 v20; // rcx
  char v21; // dl
  __int64 v22; // rdx
  unsigned int *v23; // rdi
  __int64 *v24; // r12
  __int64 v25; // rax
  __int64 v26; // rcx
  unsigned __int64 v27; // r10
  __int64 i; // r10
  __int64 v29; // rax
  __int64 *v30; // rdi
  __int64 v31; // r9
  unsigned __int64 v32; // rax
  int v33; // r11d
  unsigned __int64 v34; // r10
  unsigned __int64 v35; // rbx
  __int64 v36; // r8
  __int64 v37; // r13
  _QWORD *v38; // rax
  __int64 v39; // r14
  __int64 v40; // r12
  _QWORD *v41; // r8
  _BYTE *v42; // r9
  __int64 v43; // r10
  int v44; // r15d
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // rax
  __int64 v47; // rdi
  unsigned __int64 v48; // rax
  __int64 j; // rcx
  __int64 v50; // rsi
  unsigned int v51; // ecx
  __int64 *v52; // rdx
  __int64 v53; // rsi
  __int64 v54; // r14
  __int64 v55; // rax
  int v56; // eax
  __int64 v57; // rdi
  __int64 v58; // r15
  __int64 *v59; // r12
  _BYTE *v60; // r10
  _QWORD *v61; // r14
  __int64 (*v62)(); // rax
  __int64 v63; // rax
  __int64 v64; // rcx
  __int64 v65; // rsi
  unsigned int v66; // ebx
  __int64 *v67; // rdx
  __int64 v68; // rdi
  __int64 v69; // rcx
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r8
  _BYTE *v73; // r9
  _BYTE *v74; // r10
  int v75; // r11d
  unsigned __int64 v76; // rbx
  __int64 *v77; // rax
  _QWORD *v78; // r12
  __int64 *v79; // r14
  __int64 *v80; // rax
  int v81; // ebx
  int v82; // r12d
  unsigned int v83; // ebx
  __int64 v84; // r13
  __int64 v85; // rax
  int v86; // eax
  int v87; // edx
  int v88; // r8d
  int v89; // edx
  int v90; // esi
  __int64 v91; // rdx
  _BYTE *v92; // [rsp-108h] [rbp-108h]
  _QWORD *v93; // [rsp-108h] [rbp-108h]
  int v94; // [rsp-108h] [rbp-108h]
  unsigned __int64 v95; // [rsp-100h] [rbp-100h]
  unsigned int v96; // [rsp-100h] [rbp-100h]
  _BYTE *v97; // [rsp-100h] [rbp-100h]
  int v98; // [rsp-F8h] [rbp-F8h]
  __int64 v99; // [rsp-F8h] [rbp-F8h]
  unsigned __int64 v100; // [rsp-F0h] [rbp-F0h]
  __int64 v101; // [rsp-F0h] [rbp-F0h]
  _QWORD *v102; // [rsp-F0h] [rbp-F0h]
  __int64 v103; // [rsp-F0h] [rbp-F0h]
  int v104; // [rsp-F0h] [rbp-F0h]
  __int64 *v105; // [rsp-F0h] [rbp-F0h]
  _BYTE *v106; // [rsp-F0h] [rbp-F0h]
  _BYTE *v107; // [rsp-F0h] [rbp-F0h]
  _BYTE *v108; // [rsp-F0h] [rbp-F0h]
  int v109; // [rsp-E8h] [rbp-E8h]
  __int64 *v110; // [rsp-E8h] [rbp-E8h]
  __int64 *v111; // [rsp-E8h] [rbp-E8h]
  int v112; // [rsp-E8h] [rbp-E8h]
  int v113; // [rsp-E8h] [rbp-E8h]
  int v114; // [rsp-E8h] [rbp-E8h]
  __int64 v115; // [rsp-E0h] [rbp-E0h]
  unsigned __int8 v116; // [rsp-E0h] [rbp-E0h]
  __int64 v117; // [rsp-D8h] [rbp-D8h]
  int v118; // [rsp-D8h] [rbp-D8h]
  __int64 *v119; // [rsp-D8h] [rbp-D8h]
  __int16 v120; // [rsp-CAh] [rbp-CAh]
  __int64 v121; // [rsp-C8h] [rbp-C8h]
  unsigned __int8 v122; // [rsp-C0h] [rbp-C0h]
  __int64 v123; // [rsp-C0h] [rbp-C0h]
  __int32 v124; // [rsp-ACh] [rbp-ACh] BYREF
  __int64 v125; // [rsp-A8h] [rbp-A8h] BYREF
  __int64 v126; // [rsp-A0h] [rbp-A0h] BYREF
  unsigned __int64 v127; // [rsp-98h] [rbp-98h] BYREF
  __int64 v128; // [rsp-90h] [rbp-90h]
  __int64 v129; // [rsp-88h] [rbp-88h]
  __int64 v130; // [rsp-80h] [rbp-80h]
  char *v131; // [rsp-68h] [rbp-68h] BYREF
  __int64 v132; // [rsp-60h] [rbp-60h]
  _BYTE v133[88]; // [rsp-58h] [rbp-58h] BYREF

  if ( !a3 )
    return 0;
  v6 = 16 * a3;
  result = 0;
  v8 = *a2;
  if ( *(__int64 *)((char *)a2 + v6 - 16) == *a2 && (*(_BYTE *)(v8 + 46) & 0xC) == 0 )
  {
    v11 = **(unsigned __int16 **)(v8 + 16);
    v12 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 72) + 432LL);
    v120 = **(_WORD **)(v8 + 16);
    if ( v12 != sub_1F39450 )
    {
      v123 = v6;
      v56 = v12();
      v6 = v123;
      a6 = v56;
      if ( (_BYTE)v56 )
        goto LABEL_7;
      v11 = **(unsigned __int16 **)(v8 + 16);
    }
    v13 = v11 & 0xFFFFFFFB;
    LOBYTE(a6) = (v11 & 0xFFFB) == 19;
    LOBYTE(v13) = (_WORD)v11 == 21;
    a6 |= v13;
LABEL_7:
    v131 = v133;
    v132 = 0x800000000LL;
    if ( (__int64 *)((char *)a2 + v6) == a2 )
      return 0;
    v14 = a2;
    v15 = a2;
    v16 = 0;
    v17 = (__int64 *)((char *)a2 + v6);
    while ( 1 )
    {
      v19 = *((_DWORD *)v14 + 2);
      v20 = *(_QWORD *)(v8 + 32) + 40LL * v19;
      v21 = *(_BYTE *)(v20 + 3);
      if ( (v21 & 0x20) != 0 )
      {
        v16 = *(_DWORD *)(v20 + 8);
      }
      else
      {
        if ( !(_BYTE)a6 && (*(_DWORD *)v20 & 0xFFF00) != 0 )
          goto LABEL_23;
        v18 = v21 & 0x10;
        if ( a4 )
        {
          if ( v18 )
            goto LABEL_23;
          if ( *(_BYTE *)v20 )
          {
LABEL_20:
            v22 = (unsigned int)v132;
            if ( (unsigned int)v132 >= HIDWORD(v132) )
            {
              v105 = v15;
              v111 = v14;
              v116 = a6;
              v118 = v16;
              sub_16CD150((__int64)&v131, v133, 0, 4, (int)v15, a6);
              v22 = (unsigned int)v132;
              v15 = v105;
              v14 = v111;
              a6 = v116;
              v16 = v118;
            }
            *(_DWORD *)&v131[4 * v22] = v19;
            LODWORD(v132) = v132 + 1;
            goto LABEL_15;
          }
        }
        else if ( *(_BYTE *)v20 | v18 )
        {
          goto LABEL_20;
        }
        if ( (*(_WORD *)(v20 + 2) & 0xFF0) == 0 )
          goto LABEL_20;
      }
LABEL_15:
      v14 += 2;
      if ( v17 == v14 )
      {
        v24 = v15;
        if ( !(_DWORD)v132 )
          goto LABEL_23;
        v25 = *(_QWORD *)(v8 + 24);
        v26 = *(_QWORD *)v8;
        v115 = v25;
        if ( v8 == *(_QWORD *)(v25 + 32) )
        {
          v27 = v25 + 24;
        }
        else
        {
          if ( (v26 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
            BUG();
          v27 = v26 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_QWORD *)(v26 & 0xFFFFFFFFFFFFFFF8LL) & 4) == 0 && (*(_BYTE *)(v27 + 46) & 4) != 0 )
          {
            for ( i = *(_QWORD *)(v26 & 0xFFFFFFFFFFFFFFF8LL); ; i = *(_QWORD *)v27 )
            {
              v27 = i & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_BYTE *)(v27 + 46) & 4) == 0 )
                break;
            }
          }
        }
        v29 = v8;
        if ( (v26 & 4) == 0 && (*(_BYTE *)(v8 + 46) & 8) != 0 )
        {
          do
            v29 = *(_QWORD *)(v29 + 8);
          while ( (*(_BYTE *)(v29 + 46) & 8) != 0 );
        }
        v100 = v27;
        v30 = *(__int64 **)(a1 + 72);
        v31 = *(_QWORD *)(a1 + 16);
        v109 = v16;
        v117 = *(_QWORD *)(v29 + 8);
        if ( a4 )
        {
          v32 = sub_1F3B570(v30, v8, v131, (unsigned int)v132, a4, v31);
          v33 = v109;
          v34 = v100;
        }
        else
        {
          v32 = sub_1F3AF50(v30, v8, (unsigned int *)v131, (unsigned int)v132, *(_DWORD *)(a1 + 112), v31);
          v34 = v100;
          v33 = v109;
        }
        v121 = v32;
        if ( !v32 )
        {
LABEL_23:
          v23 = (unsigned int *)v131;
          result = 0;
          goto LABEL_24;
        }
        v35 = v8;
        if ( (*(_BYTE *)(v8 + 46) & 4) != 0 )
        {
          do
            v35 = *(_QWORD *)v35 & 0xFFFFFFFFFFFFFFF8LL;
          while ( (*(_BYTE *)(v35 + 46) & 4) != 0 );
        }
        do
        {
          v36 = *(_QWORD *)(v35 + 32);
          v37 = v36 + 40LL * *(unsigned int *)(v35 + 40);
          if ( v36 != v37 )
            break;
          v35 = *(_QWORD *)(v35 + 8);
          if ( *(_QWORD *)(v8 + 24) + 24LL == v35 )
            break;
        }
        while ( (*(_BYTE *)(v35 + 46) & 4) != 0 );
        v38 = (_QWORD *)a1;
        v110 = v24;
        v39 = v36;
        v40 = *(_QWORD *)(v8 + 24) + 24LL;
        v41 = v38;
        v42 = (_BYTE *)v34;
        v43 = v8;
        while ( 2 )
        {
          if ( v37 == v39 )
          {
            v57 = v41[9];
            v58 = v43;
            v59 = v110;
            v60 = v42;
            v61 = v41;
            v62 = *(__int64 (**)())(*(_QWORD *)v57 + 80LL);
            if ( v62 != sub_1EBAF80 )
            {
              v114 = v33;
              v108 = v42;
              v86 = ((__int64 (__fastcall *)(__int64, __int64, __int32 *))v62)(v57, v58, &v124);
              v33 = v114;
              v60 = v108;
              if ( v86 )
              {
                sub_20F2770((__int64)(v61 + 57), v58, v124);
                v60 = v108;
                v33 = v114;
              }
            }
            v63 = *(_QWORD *)(v61[2] + 272LL);
            v64 = *(unsigned int *)(v63 + 384);
            if ( (_DWORD)v64 )
            {
              v65 = *(_QWORD *)(v63 + 368);
              v66 = (v64 - 1) & (((unsigned int)v58 >> 4) ^ ((unsigned int)v58 >> 9));
              v67 = (__int64 *)(v65 + 16LL * v66);
              v68 = *v67;
              if ( v58 == *v67 )
              {
LABEL_78:
                if ( v67 != (__int64 *)(v65 + 16 * v64) )
                {
                  v69 = v67[1];
                  v106 = v60;
                  v112 = v33;
                  *(_QWORD *)((v69 & 0xFFFFFFFFFFFFFFF8LL) + 16) = v121;
                  *v67 = -16;
                  --*(_DWORD *)(v63 + 376);
                  ++*(_DWORD *)(v63 + 380);
                  v125 = v121;
                  v126 = v69;
                  sub_20EB8E0((__int64)&v127, v63 + 360, &v125, &v126);
                  v33 = v112;
                  v60 = v106;
                }
              }
              else
              {
                v87 = 1;
                while ( v68 != -8 )
                {
                  v88 = v87 + 1;
                  v66 = (v64 - 1) & (v87 + v66);
                  v67 = (__int64 *)(v65 + 16LL * v66);
                  v68 = *v67;
                  if ( v58 == *v67 )
                    goto LABEL_78;
                  v87 = v88;
                }
              }
            }
            v107 = v60;
            v113 = v33;
            sub_1E16240(v58);
            v74 = v107;
            v75 = v113;
            if ( v107 == (_BYTE *)(v115 + 24) )
            {
              v76 = *(_QWORD *)(v115 + 32);
            }
            else
            {
              if ( (*v107 & 4) == 0 && (v107[46] & 8) != 0 )
              {
                do
                  v74 = (_BYTE *)*((_QWORD *)v74 + 1);
                while ( (v74[46] & 8) != 0 );
              }
              v76 = *((_QWORD *)v74 + 1);
            }
            if ( v117 != v76 )
            {
              v77 = v59;
              v78 = v61;
              v79 = v77;
              do
              {
                if ( v121 != v76 )
                {
                  sub_1DC1550(*(_QWORD *)(v78[2] + 272LL), v76, 0);
                  if ( !v76 )
                    BUG();
                }
                if ( (*(_BYTE *)v76 & 4) == 0 )
                {
                  while ( (*(_BYTE *)(v76 + 46) & 8) != 0 )
                    v76 = *(_QWORD *)(v76 + 8);
                }
                v76 = *(_QWORD *)(v76 + 8);
              }
              while ( v76 != v117 );
              v80 = v79;
              v75 = v113;
              v61 = v78;
              v59 = v80;
            }
            if ( v75 )
            {
              v81 = *(_DWORD *)(v121 + 40);
              if ( v81 )
              {
                v119 = v59;
                v82 = v75;
                v83 = v81 - 1;
                v84 = 40LL * v83;
                while ( 1 )
                {
                  v85 = v84 + *(_QWORD *)(v121 + 32);
                  if ( *(_BYTE *)v85 || (*(_BYTE *)(v85 + 3) & 0x20) == 0 )
                    break;
                  if ( v82 == *(_DWORD *)(v85 + 8) )
                    sub_1E16C90(v121, v83, v70, v71, v72, v73);
                  v84 -= 40;
                  if ( !v83 )
                    break;
                  --v83;
                }
                v59 = v119;
              }
            }
            if ( v120 == 15 && !*((_DWORD *)v59 + 2) )
              sub_20F1CA0((__int64)(v61 + 57), v121, *((_DWORD *)v61 + 28), *((_DWORD *)v61 + 29), v72);
            v23 = (unsigned int *)v131;
            result = 1;
LABEL_24:
            if ( v23 != (unsigned int *)v133 )
            {
              v122 = result;
              _libc_free((unsigned __int64)v23);
              return v122;
            }
            return result;
          }
LABEL_48:
          if ( !*(_BYTE *)v39 )
          {
            v44 = *(_DWORD *)(v39 + 8);
            if ( v44 > 0
              && (*(_QWORD *)(*(_QWORD *)(v41[8] + 304LL) + 8LL * ((unsigned int)v44 >> 6)) & (1LL << v44)) == 0
              && (*(_BYTE *)(v39 + 3) & 0x10) != 0 )
            {
              v127 = 0;
              v128 = 0;
              v45 = v121;
              if ( (*(_BYTE *)(v121 + 46) & 4) != 0 )
              {
                do
                  v45 = *(_QWORD *)v45 & 0xFFFFFFFFFFFFFFF8LL;
                while ( (*(_BYTE *)(v45 + 46) & 4) != 0 );
              }
              v127 = v45;
              v101 = *(_QWORD *)(v121 + 24);
              v128 = v101 + 24;
              v129 = *(_QWORD *)(v45 + 32);
              v130 = v129 + 40LL * *(unsigned int *)(v45 + 40);
              if ( v129 == v130 )
              {
                do
                {
                  v45 = *(_QWORD *)(v45 + 8);
                  if ( v101 + 24 == v45 )
                  {
                    v127 = v101 + 24;
                    goto LABEL_60;
                  }
                  if ( (*(_BYTE *)(v45 + 46) & 4) == 0 )
                    break;
                  v129 = *(_QWORD *)(v45 + 32);
                  v130 = v129 + 40LL * *(unsigned int *)(v45 + 40);
                }
                while ( v129 == v130 );
                v127 = v45;
              }
LABEL_60:
              v92 = v42;
              v95 = v43;
              v98 = v33;
              v102 = v41;
              v46 = sub_1E13AC0((__int64 *)&v127, v44, v41[10]);
              v41 = v102;
              v33 = v98;
              v43 = v95;
              v42 = v92;
              if ( (v46 & 0xFF0000) == 0 )
              {
                v47 = v102[2];
                v48 = v95;
                for ( j = *(_QWORD *)(v47 + 272);
                      (*(_BYTE *)(v48 + 46) & 4) != 0;
                      v48 = *(_QWORD *)v48 & 0xFFFFFFFFFFFFFFF8LL )
                {
                  ;
                }
                v50 = *(_QWORD *)(j + 368);
                v51 = *(_DWORD *)(j + 384);
                v103 = v50;
                if ( v51 )
                {
                  v96 = (v51 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
                  v52 = (__int64 *)(v50 + 16LL * v96);
                  v53 = *v52;
                  if ( v48 == *v52 )
                    goto LABEL_65;
                  v89 = 1;
                  while ( v53 != -8 )
                  {
                    v90 = v89 + 1;
                    v91 = (v51 - 1) & (v96 + v89);
                    v96 = v91;
                    v52 = (__int64 *)(v103 + 16 * v91);
                    v94 = v90;
                    v53 = *v52;
                    if ( v48 == *v52 )
                      goto LABEL_65;
                    v89 = v94;
                  }
                }
                v52 = (__int64 *)(v103 + 16LL * v51);
LABEL_65:
                v93 = v41;
                v97 = v42;
                v99 = v43;
                v104 = v33;
                sub_1DBE8F0(v47, v44, v52[1] & 0xFFFFFFFFFFFFFFF8LL | 4);
                v41 = v93;
                v42 = v97;
                v43 = v99;
                v33 = v104;
              }
            }
          }
          v54 = v39 + 40;
          v55 = v37;
          if ( v54 == v37 )
          {
            while ( 1 )
            {
              v35 = *(_QWORD *)(v35 + 8);
              if ( v40 == v35 || (*(_BYTE *)(v35 + 46) & 4) == 0 )
                break;
              v37 = *(_QWORD *)(v35 + 32);
              v55 = v37 + 40LL * *(unsigned int *)(v35 + 40);
              if ( v37 != v55 )
                goto LABEL_72;
            }
            v39 = v37;
            v37 = v55;
            continue;
          }
          break;
        }
        v37 = v54;
LABEL_72:
        v39 = v37;
        v37 = v55;
        goto LABEL_48;
      }
    }
  }
  return result;
}
