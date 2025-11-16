// Function: sub_1B3CF30
// Address: 0x1b3cf30
//
__int64 __fastcall sub_1B3CF30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // r14
  __int64 *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r12
  __int64 v13; // r13
  __int64 *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // r13
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // r15
  __int64 v25; // r11
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // rdi
  __int64 v29; // r12
  __int64 v30; // rbx
  __int64 v31; // r14
  __int64 v32; // r14
  unsigned int v33; // r15d
  __int64 v34; // rdx
  _QWORD *v35; // rax
  __int64 v36; // rsi
  unsigned __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 *v41; // rdx
  __int64 v42; // r12
  __int64 v43; // rdx
  __int64 v44; // r13
  __int64 v45; // rax
  int v46; // edx
  __int64 *v47; // r12
  __int64 v48; // rcx
  unsigned int v49; // edi
  __int64 *v50; // rax
  __int64 v51; // rax
  __int64 v52; // r13
  __int64 v53; // r15
  __int64 v54; // r14
  unsigned int v55; // esi
  __int64 v56; // r9
  unsigned int v57; // edi
  __int64 *v58; // rax
  __int64 v59; // r8
  int v60; // r15d
  int v61; // r15d
  __int64 v62; // r10
  int v63; // eax
  __int64 *v64; // rdx
  __int64 v65; // rdx
  unsigned int v66; // ecx
  __int64 v67; // rbx
  int v68; // r11d
  int v69; // eax
  int v70; // r11d
  int v71; // r11d
  unsigned int v72; // r15d
  int v73; // edi
  __int64 *v74; // r11
  int v75; // edi
  int v76; // edi
  int v77; // eax
  int v78; // r10d
  __int64 v79; // r11
  unsigned int v80; // edx
  __int64 v81; // r9
  int v82; // r8d
  __int64 *v83; // rsi
  int v84; // eax
  int v85; // r10d
  __int64 v86; // r11
  int v87; // r8d
  unsigned int v88; // edx
  __int64 v89; // r9
  int v90; // r11d
  __int64 *v91; // r15
  int v92; // edi
  int v93; // ecx
  __int64 v94; // rdx
  int v95; // eax
  int v96; // r11d
  __int64 v97; // r15
  __int64 *v98; // rdi
  int v99; // eax
  int v100; // r11d
  __int64 v101; // r15
  int v102; // r11d
  int v103; // r11d
  __int64 v104; // r10
  int v105; // edx
  unsigned int v106; // ecx
  __int64 v107; // rdx
  __int64 *v108; // rdi
  int v109; // edi
  int v110; // eax
  int v111; // r13d
  __int64 v112; // r11
  __int64 *v113; // rcx
  __int64 v114; // rdi
  int v115; // r10d
  __int64 v116; // [rsp+0h] [rbp-90h]
  __int64 v117; // [rsp+8h] [rbp-88h]
  __int64 v118; // [rsp+10h] [rbp-80h]
  __int64 v119; // [rsp+10h] [rbp-80h]
  __int64 v120; // [rsp+10h] [rbp-80h]
  int v121; // [rsp+10h] [rbp-80h]
  __int64 v122; // [rsp+10h] [rbp-80h]
  int v123; // [rsp+18h] [rbp-78h]
  __int64 v124; // [rsp+18h] [rbp-78h]
  __int64 **v126; // [rsp+28h] [rbp-68h]
  __int64 *v127; // [rsp+30h] [rbp-60h]
  __int64 v128; // [rsp+30h] [rbp-60h]
  __int64 v129; // [rsp+30h] [rbp-60h]
  __int64 v130; // [rsp+30h] [rbp-60h]
  __int64 **v131; // [rsp+38h] [rbp-58h]
  __int64 *v132; // [rsp+38h] [rbp-58h]
  __int64 v133; // [rsp+40h] [rbp-50h] BYREF
  __int16 v134; // [rsp+50h] [rbp-40h]

  v117 = a1 + 24;
  result = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  v131 = *(__int64 ***)a2;
  v126 = (__int64 **)result;
  if ( result != *(_QWORD *)a2 )
  {
    v8 = a2;
LABEL_4:
    while ( 1 )
    {
      v9 = *v131;
      v127 = v9;
      if ( (__int64 *)v9[2] == v9 )
        break;
LABEL_3:
      if ( v126 == ++v131 )
        goto LABEL_22;
    }
    v10 = sub_157F280(*v9);
    v12 = v11;
    v13 = v10;
    if ( v10 == v11 )
      goto LABEL_14;
    while ( 1 )
    {
      a2 = v13;
      if ( (unsigned __int8)sub_1B3C810(a1, v13) )
        break;
      v14 = *(__int64 **)v8;
      v15 = *(_QWORD *)v8 + 8LL * *(unsigned int *)(v8 + 8);
      if ( *(_QWORD *)v8 != v15 )
      {
        do
        {
          v16 = *v14++;
          *(_QWORD *)(v16 + 56) = 0;
        }
        while ( (__int64 *)v15 != v14 );
      }
      if ( !v13 )
        BUG();
      v17 = *(_QWORD *)(v13 + 32);
      if ( !v17 )
        BUG();
      v13 = 0;
      if ( *(_BYTE *)(v17 - 8) == 77 )
        v13 = v17 - 24;
      if ( v12 == v13 )
        goto LABEL_14;
    }
    v47 = *(__int64 **)v8;
    if ( *(_QWORD *)v8 == *(_QWORD *)v8 + 8LL * *(unsigned int *)(v8 + 8) )
    {
LABEL_14:
      v18 = (__int64 *)v127[1];
      if ( !v18 )
        goto LABEL_15;
      goto LABEL_3;
    }
    v124 = v8;
    v48 = *(_QWORD *)v8 + 8LL * *(unsigned int *)(v8 + 8);
    while ( 1 )
    {
      v52 = *(_QWORD *)(*v47 + 56);
      if ( !v52 )
        goto LABEL_52;
      v53 = *(_QWORD *)(a1 + 8);
      v54 = *(_QWORD *)(v52 + 40);
      v55 = *(_DWORD *)(v53 + 24);
      if ( v55 )
      {
        v56 = *(_QWORD *)(v53 + 8);
        v57 = (v55 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
        v58 = (__int64 *)(v56 + 16LL * v57);
        v59 = *v58;
        if ( v54 == *v58 )
          goto LABEL_56;
        v121 = 1;
        v74 = 0;
        while ( v59 != -8 )
        {
          if ( !v74 && v59 == -16 )
            v74 = v58;
          v57 = (v55 - 1) & (v121 + v57);
          v58 = (__int64 *)(v56 + 16LL * v57);
          v59 = *v58;
          if ( v54 == *v58 )
            goto LABEL_56;
          ++v121;
        }
        v75 = *(_DWORD *)(v53 + 16);
        if ( v74 )
          v58 = v74;
        ++*(_QWORD *)v53;
        v76 = v75 + 1;
        if ( 4 * v76 < 3 * v55 )
        {
          if ( v55 - *(_DWORD *)(v53 + 20) - v76 > v55 >> 3 )
            goto LABEL_93;
          v116 = v48;
          sub_141A900(v53, v55);
          v84 = *(_DWORD *)(v53 + 24);
          if ( !v84 )
          {
LABEL_222:
            ++*(_DWORD *)(v53 + 16);
            BUG();
          }
          v85 = v84 - 1;
          v86 = *(_QWORD *)(v53 + 8);
          v83 = 0;
          v48 = v116;
          v87 = 1;
          v88 = (v84 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
          v76 = *(_DWORD *)(v53 + 16) + 1;
          v58 = (__int64 *)(v86 + 16LL * v88);
          v89 = *v58;
          if ( v54 == *v58 )
            goto LABEL_93;
          while ( v89 != -8 )
          {
            if ( !v83 && v89 == -16 )
              v83 = v58;
            v88 = v85 & (v87 + v88);
            v58 = (__int64 *)(v86 + 16LL * v88);
            v89 = *v58;
            if ( v54 == *v58 )
              goto LABEL_93;
            ++v87;
          }
          goto LABEL_101;
        }
      }
      else
      {
        ++*(_QWORD *)v53;
      }
      v122 = v48;
      sub_141A900(v53, 2 * v55);
      v77 = *(_DWORD *)(v53 + 24);
      if ( !v77 )
        goto LABEL_222;
      v78 = v77 - 1;
      v79 = *(_QWORD *)(v53 + 8);
      v48 = v122;
      v80 = (v77 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
      v76 = *(_DWORD *)(v53 + 16) + 1;
      v58 = (__int64 *)(v79 + 16LL * v80);
      v81 = *v58;
      if ( v54 == *v58 )
        goto LABEL_93;
      v82 = 1;
      v83 = 0;
      while ( v81 != -8 )
      {
        if ( !v83 && v81 == -16 )
          v83 = v58;
        v80 = v78 & (v82 + v80);
        v58 = (__int64 *)(v79 + 16LL * v80);
        v81 = *v58;
        if ( v54 == *v58 )
          goto LABEL_93;
        ++v82;
      }
LABEL_101:
      if ( v83 )
        v58 = v83;
LABEL_93:
      *(_DWORD *)(v53 + 16) = v76;
      if ( *v58 != -8 )
        --*(_DWORD *)(v53 + 20);
      *v58 = v54;
      v58[1] = 0;
LABEL_56:
      v58[1] = v52;
      a2 = *(unsigned int *)(a1 + 48);
      if ( !(_DWORD)a2 )
      {
        ++*(_QWORD *)(a1 + 24);
        goto LABEL_58;
      }
      a6 = *(_QWORD *)(a1 + 32);
      v49 = (a2 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
      v50 = (__int64 *)(a6 + 16LL * v49);
      a5 = *v50;
      if ( v54 != *v50 )
      {
        v68 = 1;
        v64 = 0;
        while ( a5 != -8 )
        {
          if ( a5 == -16 && !v64 )
            v64 = v50;
          v49 = (a2 - 1) & (v68 + v49);
          v50 = (__int64 *)(a6 + 16LL * v49);
          a5 = *v50;
          if ( v54 == *v50 )
            goto LABEL_50;
          ++v68;
        }
        if ( !v64 )
          v64 = v50;
        v69 = *(_DWORD *)(a1 + 40);
        ++*(_QWORD *)(a1 + 24);
        v63 = v69 + 1;
        if ( 4 * v63 >= (unsigned int)(3 * a2) )
        {
LABEL_58:
          v119 = v48;
          sub_1B3C650(v117, 2 * a2);
          v60 = *(_DWORD *)(a1 + 48);
          if ( !v60 )
            goto LABEL_221;
          v61 = v60 - 1;
          v62 = *(_QWORD *)(a1 + 32);
          v48 = v119;
          a2 = v61 & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
          v63 = *(_DWORD *)(a1 + 40) + 1;
          v64 = (__int64 *)(v62 + 16 * a2);
          a6 = *v64;
          if ( v54 != *v64 )
          {
            a5 = 1;
            v108 = 0;
            while ( a6 != -8 )
            {
              if ( !v108 && a6 == -16 )
                v108 = v64;
              a2 = v61 & (unsigned int)(a5 + a2);
              v64 = (__int64 *)(v62 + 16LL * (unsigned int)a2);
              a6 = *v64;
              if ( v54 == *v64 )
                goto LABEL_60;
              a5 = (unsigned int)(a5 + 1);
            }
            if ( v108 )
              v64 = v108;
          }
        }
        else
        {
          a5 = (unsigned int)a2 >> 3;
          if ( (int)a2 - *(_DWORD *)(a1 + 44) - v63 <= (unsigned int)a5 )
          {
            v120 = v48;
            sub_1B3C650(v117, a2);
            v70 = *(_DWORD *)(a1 + 48);
            if ( !v70 )
            {
LABEL_221:
              ++*(_DWORD *)(a1 + 40);
              BUG();
            }
            v71 = v70 - 1;
            a6 = *(_QWORD *)(a1 + 32);
            a2 = 0;
            v72 = v71 & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
            v48 = v120;
            v73 = 1;
            v63 = *(_DWORD *)(a1 + 40) + 1;
            v64 = (__int64 *)(a6 + 16LL * v72);
            a5 = *v64;
            if ( v54 != *v64 )
            {
              while ( a5 != -8 )
              {
                if ( a5 == -16 && !a2 )
                  a2 = (__int64)v64;
                v72 = v71 & (v73 + v72);
                v64 = (__int64 *)(a6 + 16LL * v72);
                a5 = *v64;
                if ( v54 == *v64 )
                  goto LABEL_60;
                ++v73;
              }
              if ( a2 )
                v64 = (__int64 *)a2;
            }
          }
        }
LABEL_60:
        *(_DWORD *)(a1 + 40) = v63;
        if ( *v64 != -8 )
          --*(_DWORD *)(a1 + 44);
        *v64 = v54;
        v51 = 0;
        v64[1] = 0;
        goto LABEL_51;
      }
LABEL_50:
      v51 = v50[1];
LABEL_51:
      *(_QWORD *)(v51 + 8) = v52;
LABEL_52:
      if ( (__int64 *)v48 == ++v47 )
      {
        v8 = v124;
        v18 = (__int64 *)v127[1];
        if ( v18 )
          goto LABEL_3;
LABEL_15:
        v19 = *(_QWORD *)a1;
        v20 = *(_QWORD *)(*v127 + 48);
        v123 = *((_DWORD *)v127 + 10);
        v134 = 260;
        v21 = *(_QWORD *)(v19 + 8);
        if ( v20 )
          v20 -= 24;
        v133 = v19 + 16;
        v118 = v20;
        v22 = sub_1648B60(64);
        v23 = v22;
        if ( v22 )
        {
          sub_15F1EA0(v22, v21, 53, 0, 0, v118);
          *(_DWORD *)(v23 + 56) = v123;
          sub_164B780(v23, &v133);
          sub_1648880(v23, *(_DWORD *)(v23 + 56), 1);
        }
        v127[1] = v23;
        v24 = *(_QWORD *)(a1 + 8);
        a2 = *(unsigned int *)(v24 + 24);
        if ( !(_DWORD)a2 )
        {
          ++*(_QWORD *)v24;
LABEL_138:
          sub_141A900(v24, 2 * a2);
          v102 = *(_DWORD *)(v24 + 24);
          if ( v102 )
          {
            v103 = v102 - 1;
            v104 = *(_QWORD *)(v24 + 8);
            v105 = *(_DWORD *)(v24 + 16) + 1;
            v106 = v103 & (((unsigned int)*v127 >> 9) ^ ((unsigned int)*v127 >> 4));
            v27 = (__int64 *)(v104 + 16LL * v106);
            a2 = *v27;
            if ( *v127 != *v27 )
            {
              a6 = 1;
              while ( a2 != -8 )
              {
                if ( !v18 && a2 == -16 )
                  v18 = v27;
                a5 = (unsigned int)(a6 + 1);
                v106 = v103 & (a6 + v106);
                v27 = (__int64 *)(v104 + 16LL * v106);
                a2 = *v27;
                if ( *v127 == *v27 )
                  goto LABEL_140;
                a6 = (unsigned int)a5;
              }
              if ( v18 )
                v27 = v18;
            }
            goto LABEL_140;
          }
          goto LABEL_219;
        }
        v25 = *(_QWORD *)(v24 + 8);
        v26 = (a2 - 1) & (((unsigned int)*v127 >> 9) ^ ((unsigned int)*v127 >> 4));
        v27 = (__int64 *)(v25 + 16LL * v26);
        v28 = *v27;
        if ( *v27 == *v127 )
          goto LABEL_21;
        a6 = 0;
        a5 = 1;
        while ( v28 != -8 )
        {
          if ( v28 == -16 && !a6 )
            a6 = (__int64)v27;
          v26 = (a2 - 1) & (a5 + v26);
          v27 = (__int64 *)(v25 + 16LL * v26);
          v28 = *v27;
          if ( *v127 == *v27 )
            goto LABEL_21;
          a5 = (unsigned int)(a5 + 1);
        }
        v109 = *(_DWORD *)(v24 + 16);
        if ( a6 )
          v27 = (__int64 *)a6;
        ++*(_QWORD *)v24;
        v105 = v109 + 1;
        if ( 4 * (v109 + 1) >= (unsigned int)(3 * a2) )
          goto LABEL_138;
        if ( (int)a2 - *(_DWORD *)(v24 + 20) - v105 <= (unsigned int)a2 >> 3 )
        {
          sub_141A900(v24, a2);
          v110 = *(_DWORD *)(v24 + 24);
          if ( v110 )
          {
            v111 = v110 - 1;
            v112 = *(_QWORD *)(v24 + 8);
            a5 = *v127;
            a2 = (v110 - 1) & (((unsigned int)*v127 >> 9) ^ ((unsigned int)*v127 >> 4));
            v105 = *(_DWORD *)(v24 + 16) + 1;
            v113 = (__int64 *)(v112 + 16 * a2);
            v114 = *v113;
            v27 = v113;
            if ( *v113 != *v127 )
            {
              v27 = 0;
              v115 = 1;
              while ( v114 != -8 )
              {
                if ( v114 == -16 && !v27 )
                  v27 = v113;
                a6 = (unsigned int)(v115 + 1);
                a2 = v111 & (unsigned int)(v115 + a2);
                v113 = (__int64 *)(v112 + 16LL * (unsigned int)a2);
                v114 = *v113;
                if ( a5 == *v113 )
                {
                  v27 = (__int64 *)(v112 + 16LL * (unsigned int)a2);
                  goto LABEL_140;
                }
                ++v115;
              }
              if ( !v27 )
                v27 = v113;
            }
            goto LABEL_140;
          }
LABEL_219:
          ++*(_DWORD *)(v24 + 16);
          BUG();
        }
LABEL_140:
        *(_DWORD *)(v24 + 16) = v105;
        if ( *v27 != -8 )
          --*(_DWORD *)(v24 + 20);
        v107 = *v127;
        v27[1] = 0;
        *v27 = v107;
LABEL_21:
        ++v131;
        v27[1] = v23;
        if ( v126 == v131 )
        {
LABEL_22:
          result = *(unsigned int *)(v8 + 8);
          v29 = *(_QWORD *)v8 + 8 * result;
          v132 = *(__int64 **)v8;
          if ( *(_QWORD *)v8 == v29 )
            return result;
          while ( 2 )
          {
            v30 = *(_QWORD *)(v29 - 8);
            v31 = *(_QWORD *)(v30 + 16);
            if ( v31 == v30 )
            {
              v32 = *(_QWORD *)(v30 + 8);
              if ( *(_BYTE *)(v32 + 16) == 77 )
              {
                result = *(_DWORD *)(v32 + 20) & 0xFFFFFFF;
                if ( (*(_DWORD *)(v32 + 20) & 0xFFFFFFF) == 0 )
                {
                  if ( *(_DWORD *)(v30 + 40) )
                  {
                    v128 = v29;
                    v33 = 0;
                    while ( 1 )
                    {
                      v40 = *(_QWORD *)(v30 + 48);
                      v41 = *(__int64 **)(v40 + 8LL * v33);
                      v42 = *v41;
                      v43 = v41[2];
                      v44 = *(_QWORD *)(v43 + 8);
                      if ( (_DWORD)result == *(_DWORD *)(v32 + 56) )
                      {
                        sub_15F55D0(v32, a2, v43, v40, a5, a6);
                        LODWORD(result) = *(_DWORD *)(v32 + 20) & 0xFFFFFFF;
                      }
                      v45 = ((_DWORD)result + 1) & 0xFFFFFFF;
                      v46 = v45 | *(_DWORD *)(v32 + 20) & 0xF0000000;
                      *(_DWORD *)(v32 + 20) = v46;
                      if ( (v46 & 0x40000000) != 0 )
                        v34 = *(_QWORD *)(v32 - 8);
                      else
                        v34 = v32 - 24 * v45;
                      v35 = (_QWORD *)(v34 + 24LL * (unsigned int)(v45 - 1));
                      if ( *v35 )
                      {
                        v36 = v35[1];
                        v37 = v35[2] & 0xFFFFFFFFFFFFFFFCLL;
                        *(_QWORD *)v37 = v36;
                        if ( v36 )
                          *(_QWORD *)(v36 + 16) = *(_QWORD *)(v36 + 16) & 3LL | v37;
                      }
                      *v35 = v44;
                      if ( v44 )
                      {
                        v38 = *(_QWORD *)(v44 + 8);
                        v35[1] = v38;
                        if ( v38 )
                        {
                          a5 = (__int64)(v35 + 1);
                          *(_QWORD *)(v38 + 16) = (unsigned __int64)(v35 + 1) | *(_QWORD *)(v38 + 16) & 3LL;
                        }
                        v35[2] = (v44 + 8) | v35[2] & 3LL;
                        *(_QWORD *)(v44 + 8) = v35;
                      }
                      v39 = *(_DWORD *)(v32 + 20) & 0xFFFFFFF;
                      a2 = (*(_BYTE *)(v32 + 23) & 0x40) != 0 ? *(_QWORD *)(v32 - 8) : v32 - 24 * v39;
                      ++v33;
                      *(_QWORD *)(a2 + 8LL * (unsigned int)(v39 - 1) + 24LL * *(unsigned int *)(v32 + 56) + 8) = v42;
                      if ( *(_DWORD *)(v30 + 40) == v33 )
                        break;
                      LODWORD(result) = *(_DWORD *)(v32 + 20) & 0xFFFFFFF;
                    }
                    v29 = v128;
                  }
                  result = a1;
                  v67 = *(_QWORD *)(a1 + 16);
                  if ( v67 )
                  {
                    result = *(unsigned int *)(v67 + 8);
                    if ( (unsigned int)result >= *(_DWORD *)(v67 + 12) )
                    {
                      a2 = v67 + 16;
                      sub_16CD150(*(_QWORD *)(a1 + 16), (const void *)(v67 + 16), 0, 8, a5, a6);
                      result = *(unsigned int *)(v67 + 8);
                    }
                    *(_QWORD *)(*(_QWORD *)v67 + 8 * result) = v32;
                    ++*(_DWORD *)(v67 + 8);
                  }
                }
              }
LABEL_25:
              v29 -= 8;
              if ( v132 == (__int64 *)v29 )
                return result;
              continue;
            }
            break;
          }
          if ( *(_DWORD *)(v30 + 40) <= 1u )
            goto LABEL_25;
          v65 = *(_QWORD *)(a1 + 8);
          a2 = *(unsigned int *)(v65 + 24);
          if ( (_DWORD)a2 )
          {
            a6 = *(_QWORD *)(v65 + 8);
            v66 = (a2 - 1) & (((unsigned int)*(_QWORD *)v30 >> 9) ^ ((unsigned int)*(_QWORD *)v30 >> 4));
            result = a6 + 16LL * v66;
            a5 = *(_QWORD *)result;
            if ( *(_QWORD *)result == *(_QWORD *)v30 )
            {
LABEL_68:
              *(_QWORD *)(result + 8) = *(_QWORD *)(v31 + 8);
              goto LABEL_25;
            }
            v90 = 1;
            v91 = 0;
            while ( a5 != -8 )
            {
              if ( !v91 && a5 == -16 )
                v91 = (__int64 *)result;
              v66 = (a2 - 1) & (v90 + v66);
              result = a6 + 16LL * v66;
              a5 = *(_QWORD *)result;
              if ( *(_QWORD *)v30 == *(_QWORD *)result )
                goto LABEL_68;
              ++v90;
            }
            v92 = *(_DWORD *)(v65 + 16);
            if ( v91 )
              result = (__int64)v91;
            ++*(_QWORD *)v65;
            v93 = v92 + 1;
            if ( 4 * (v92 + 1) < (unsigned int)(3 * a2) )
            {
              a5 = (unsigned int)a2 >> 3;
              if ( (int)a2 - *(_DWORD *)(v65 + 20) - v93 <= (unsigned int)a5 )
              {
                v130 = v65;
                sub_141A900(v65, a2);
                v65 = v130;
                v99 = *(_DWORD *)(v130 + 24);
                if ( !v99 )
                {
LABEL_218:
                  ++*(_DWORD *)(v65 + 16);
                  BUG();
                }
                v100 = v99 - 1;
                a5 = 1;
                v101 = *(_QWORD *)(v130 + 8);
                v93 = *(_DWORD *)(v130 + 16) + 1;
                v98 = 0;
                a2 = (v99 - 1) & (((unsigned int)*(_QWORD *)v30 >> 9) ^ ((unsigned int)*(_QWORD *)v30 >> 4));
                result = v101 + 16 * a2;
                a6 = *(_QWORD *)result;
                if ( *(_QWORD *)v30 != *(_QWORD *)result )
                {
                  while ( a6 != -8 )
                  {
                    if ( !v98 && a6 == -16 )
                      v98 = (__int64 *)result;
                    a2 = v100 & (unsigned int)(a5 + a2);
                    result = v101 + 16LL * (unsigned int)a2;
                    a6 = *(_QWORD *)result;
                    if ( *(_QWORD *)v30 == *(_QWORD *)result )
                      goto LABEL_118;
                    a5 = (unsigned int)(a5 + 1);
                  }
                  goto LABEL_134;
                }
              }
              goto LABEL_118;
            }
          }
          else
          {
            ++*(_QWORD *)v65;
          }
          v129 = v65;
          sub_141A900(v65, 2 * a2);
          v65 = v129;
          v95 = *(_DWORD *)(v129 + 24);
          if ( !v95 )
            goto LABEL_218;
          v96 = v95 - 1;
          v97 = *(_QWORD *)(v129 + 8);
          v93 = *(_DWORD *)(v129 + 16) + 1;
          a2 = (v95 - 1) & (((unsigned int)*(_QWORD *)v30 >> 9) ^ ((unsigned int)*(_QWORD *)v30 >> 4));
          result = v97 + 16 * a2;
          a6 = *(_QWORD *)result;
          if ( *(_QWORD *)result != *(_QWORD *)v30 )
          {
            a5 = 1;
            v98 = 0;
            while ( a6 != -8 )
            {
              if ( a6 == -16 && !v98 )
                v98 = (__int64 *)result;
              a2 = v96 & (unsigned int)(a5 + a2);
              result = v97 + 16LL * (unsigned int)a2;
              a6 = *(_QWORD *)result;
              if ( *(_QWORD *)v30 == *(_QWORD *)result )
                goto LABEL_118;
              a5 = (unsigned int)(a5 + 1);
            }
LABEL_134:
            if ( v98 )
              result = (__int64)v98;
          }
LABEL_118:
          *(_DWORD *)(v65 + 16) = v93;
          if ( *(_QWORD *)result != -8 )
            --*(_DWORD *)(v65 + 20);
          v94 = *(_QWORD *)v30;
          *(_QWORD *)(result + 8) = 0;
          *(_QWORD *)result = v94;
          goto LABEL_68;
        }
        goto LABEL_4;
      }
    }
  }
  return result;
}
