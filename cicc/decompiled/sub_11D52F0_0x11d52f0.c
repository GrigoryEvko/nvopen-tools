// Function: sub_11D52F0
// Address: 0x11d52f0
//
__int64 __fastcall sub_11D52F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r14
  __int64 result; // rax
  __int64 v9; // r12
  int v10; // edx
  __int64 v11; // rcx
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // r14
  __int64 v19; // r11
  __int64 v20; // r13
  __int64 v21; // rbx
  __int64 v22; // r13
  __int64 v23; // r15
  unsigned int v24; // esi
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r13
  __int64 v31; // r12
  __int64 v32; // rax
  __int64 v33; // r11
  __int64 v34; // rbx
  __int64 v35; // r9
  unsigned int v36; // edi
  __int64 *v37; // rax
  __int64 v38; // r8
  int v39; // r10d
  __int64 *v40; // rdx
  unsigned int v41; // ecx
  __int64 *v42; // rax
  __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // r12
  __int64 v46; // r14
  __int64 v47; // r13
  unsigned int v48; // esi
  int v49; // eax
  int v50; // r8d
  __int64 v51; // r10
  unsigned int v52; // ecx
  int v53; // eax
  __int64 *v54; // rdx
  __int64 v55; // rdi
  int v56; // eax
  int v57; // r14d
  __int64 v58; // r10
  unsigned int v59; // ecx
  int v60; // eax
  int v61; // edi
  __int64 v62; // rax
  int v63; // r13d
  __int64 v64; // rbx
  __int64 v65; // rax
  __int64 v66; // r11
  _QWORD *v67; // r10
  __int64 v68; // rax
  unsigned int v69; // r14d
  __int64 v70; // r12
  int v71; // eax
  unsigned int v72; // edi
  _QWORD *v73; // rax
  __int64 v74; // rdi
  __int64 v75; // rdi
  __int64 *v76; // rdi
  __int64 v77; // r15
  __int64 v78; // rbx
  __int64 v79; // rbx
  __int64 *v80; // rcx
  int v81; // eax
  int v82; // eax
  __int64 v83; // rax
  int v84; // eax
  int v85; // eax
  int v86; // r8d
  __int64 v87; // r10
  __int64 *v88; // rsi
  int v89; // r9d
  unsigned int v90; // ecx
  __int64 v91; // rdi
  int v92; // eax
  int v93; // eax
  __int64 v94; // r10
  unsigned int v95; // edx
  int v96; // edi
  __int64 *v97; // rsi
  int v98; // eax
  __int64 v99; // r10
  int v100; // edi
  unsigned int v101; // edx
  int v102; // r10d
  int v103; // r10d
  __int64 *v104; // rcx
  unsigned int v105; // r14d
  __int64 v106; // rdi
  int v107; // r9d
  __int64 v108; // [rsp+0h] [rbp-130h]
  __int64 v109; // [rsp+8h] [rbp-128h]
  __int64 v110; // [rsp+8h] [rbp-128h]
  int v111; // [rsp+8h] [rbp-128h]
  unsigned int v112; // [rsp+8h] [rbp-128h]
  __int64 v113; // [rsp+8h] [rbp-128h]
  __int64 v114; // [rsp+10h] [rbp-120h]
  _QWORD *v115; // [rsp+10h] [rbp-120h]
  __int64 v116; // [rsp+20h] [rbp-110h]
  __int64 *v117; // [rsp+28h] [rbp-108h]
  __int64 v118; // [rsp+28h] [rbp-108h]
  __int64 v119; // [rsp+28h] [rbp-108h]
  __int64 v120; // [rsp+30h] [rbp-100h]
  __int64 v121; // [rsp+30h] [rbp-100h]
  __int64 v122; // [rsp+38h] [rbp-F8h]
  __int64 v123; // [rsp+38h] [rbp-F8h]
  __int64 v124; // [rsp+40h] [rbp-F0h]
  __int64 v125; // [rsp+40h] [rbp-F0h]
  __int64 v126; // [rsp+40h] [rbp-F0h]
  __int64 *v127; // [rsp+48h] [rbp-E8h]
  __int64 v128; // [rsp+48h] [rbp-E8h]
  int v129; // [rsp+48h] [rbp-E8h]
  int v130; // [rsp+48h] [rbp-E8h]
  int v131; // [rsp+48h] [rbp-E8h]
  __int64 v132; // [rsp+48h] [rbp-E8h]
  const char *v133[2]; // [rsp+50h] [rbp-E0h] BYREF
  _BYTE v134[16]; // [rsp+60h] [rbp-D0h] BYREF
  __int16 v135; // [rsp+70h] [rbp-C0h]

  v6 = *(__int64 **)a2;
  v122 = a2;
  result = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  v127 = (__int64 *)result;
  if ( result == *(_QWORD *)a2 )
    return result;
  while ( 1 )
  {
LABEL_4:
    v9 = *v6;
    if ( *(_QWORD *)(v9 + 16) != v9 )
      goto LABEL_3;
    v10 = *(_DWORD *)(v9 + 40);
    if ( !v10 )
      break;
    v11 = *(_QWORD *)(v9 + 48);
    v12 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v11 + 16LL) + 8LL);
    if ( !v12 )
      break;
    if ( v10 != 1 )
    {
      v13 = v11 + 8;
      v14 = v11 + 8LL * (unsigned int)(v10 - 2) + 16;
      do
      {
        v15 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v13 + 16LL) + 8LL);
        if ( !v15 || v12 != v15 )
          goto LABEL_23;
        v13 += 8;
      }
      while ( v14 != v13 );
    }
    a2 = *v6++;
    *sub_11D31A0(*(_QWORD *)(a1 + 8), (__int64 *)a2) = v12;
    v16 = *(_QWORD *)(v9 + 48);
    *(_QWORD *)(v9 + 8) = v12;
    *(_QWORD *)(v9 + 16) = *(_QWORD *)(*(_QWORD *)v16 + 16LL);
    if ( v127 == v6 )
      goto LABEL_13;
  }
LABEL_23:
  v27 = *(_QWORD *)v9;
  v133[0] = v134;
  v133[1] = (const char *)0x1400000000LL;
  v28 = sub_AA5930(v27);
  if ( v28 == v29 )
    goto LABEL_56;
  v120 = v9;
  v30 = v29;
  v31 = v28;
  while ( 1 )
  {
    a2 = v31;
    if ( (unsigned __int8)sub_11D4AA0(a1, v31, (__int64)v133) )
      break;
    if ( !v31 )
      BUG();
    v32 = *(_QWORD *)(v31 + 32);
    if ( !v32 )
      BUG();
    v31 = 0;
    if ( *(_BYTE *)(v32 - 24) == 84 )
      v31 = v32 - 24;
    if ( v30 == v31 )
    {
      v9 = v120;
      goto LABEL_56;
    }
  }
  v9 = v120;
  v33 = *(_QWORD *)v122 + 8LL * *(unsigned int *)(v122 + 8);
  if ( *(_QWORD *)v122 == v33 )
    goto LABEL_56;
  v117 = v6;
  v34 = *(_QWORD *)v122;
  v114 = a1 + 24;
  do
  {
    v45 = *(_QWORD *)(*(_QWORD *)v34 + 56LL);
    if ( v45 )
    {
      v46 = *(_QWORD *)(a1 + 8);
      v47 = *(_QWORD *)(v45 + 40);
      v48 = *(_DWORD *)(v46 + 24);
      if ( v48 )
      {
        v35 = *(_QWORD *)(v46 + 8);
        v36 = (v48 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
        v37 = (__int64 *)(v35 + 16LL * v36);
        v38 = *v37;
        if ( v47 == *v37 )
        {
LABEL_34:
          v37[1] = v45;
          a2 = *(unsigned int *)(a1 + 48);
          if ( !(_DWORD)a2 )
          {
LABEL_47:
            ++*(_QWORD *)(a1 + 24);
            goto LABEL_48;
          }
LABEL_35:
          a6 = (unsigned int)(a2 - 1);
          a5 = *(_QWORD *)(a1 + 32);
          v39 = 1;
          v40 = 0;
          v41 = a6 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
          v42 = (__int64 *)(a5 + 16LL * v41);
          v43 = *v42;
          if ( v47 == *v42 )
          {
LABEL_36:
            v44 = v42[1];
LABEL_37:
            *(_QWORD *)(v44 + 8) = v45;
            goto LABEL_38;
          }
          while ( v43 != -4096 )
          {
            if ( !v40 && v43 == -8192 )
              v40 = v42;
            v41 = a6 & (v39 + v41);
            v42 = (__int64 *)(a5 + 16LL * v41);
            v43 = *v42;
            if ( v47 == *v42 )
              goto LABEL_36;
            ++v39;
          }
          if ( !v40 )
            v40 = v42;
          v92 = *(_DWORD *)(a1 + 40);
          ++*(_QWORD *)(a1 + 24);
          v60 = v92 + 1;
          if ( 4 * v60 < (unsigned int)(3 * a2) )
          {
            if ( (int)a2 - *(_DWORD *)(a1 + 44) - v60 <= (unsigned int)a2 >> 3 )
            {
              v113 = v33;
              sub_11D3880(v114, a2);
              v102 = *(_DWORD *)(a1 + 48);
              if ( !v102 )
              {
LABEL_178:
                ++*(_DWORD *)(a1 + 40);
                BUG();
              }
              v103 = v102 - 1;
              a6 = *(_QWORD *)(a1 + 32);
              v104 = 0;
              v105 = v103 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
              v33 = v113;
              a2 = 1;
              v60 = *(_DWORD *)(a1 + 40) + 1;
              v40 = (__int64 *)(a6 + 16LL * v105);
              v106 = *v40;
              if ( v47 != *v40 )
              {
                while ( v106 != -4096 )
                {
                  if ( !v104 && v106 == -8192 )
                    v104 = v40;
                  a5 = (unsigned int)(a2 + 1);
                  v105 = v103 & (a2 + v105);
                  v40 = (__int64 *)(a6 + 16LL * v105);
                  v106 = *v40;
                  if ( v47 == *v40 )
                    goto LABEL_113;
                  a2 = (unsigned int)a5;
                }
                if ( v104 )
                  v40 = v104;
              }
            }
            goto LABEL_113;
          }
LABEL_48:
          a2 = (unsigned int)(2 * a2);
          v110 = v33;
          sub_11D3880(v114, a2);
          v56 = *(_DWORD *)(a1 + 48);
          if ( !v56 )
            goto LABEL_178;
          v57 = v56 - 1;
          v58 = *(_QWORD *)(a1 + 32);
          v33 = v110;
          v59 = (v56 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
          v60 = *(_DWORD *)(a1 + 40) + 1;
          v40 = (__int64 *)(v58 + 16LL * v59);
          a5 = *v40;
          if ( v47 != *v40 )
          {
            v61 = 1;
            a2 = 0;
            while ( a5 != -4096 )
            {
              if ( !a2 && a5 == -8192 )
                a2 = (__int64)v40;
              a6 = (unsigned int)(v61 + 1);
              v59 = v57 & (v61 + v59);
              v40 = (__int64 *)(v58 + 16LL * v59);
              a5 = *v40;
              if ( v47 == *v40 )
                goto LABEL_113;
              ++v61;
            }
            if ( a2 )
              v40 = (__int64 *)a2;
          }
LABEL_113:
          *(_DWORD *)(a1 + 40) = v60;
          if ( *v40 != -4096 )
            --*(_DWORD *)(a1 + 44);
          *v40 = v47;
          v44 = 0;
          v40[1] = 0;
          goto LABEL_37;
        }
        v111 = 1;
        v54 = 0;
        while ( v38 != -4096 )
        {
          if ( v38 == -8192 && !v54 )
            v54 = v37;
          v36 = (v48 - 1) & (v111 + v36);
          v37 = (__int64 *)(v35 + 16LL * v36);
          v38 = *v37;
          if ( v47 == *v37 )
            goto LABEL_34;
          ++v111;
        }
        if ( !v54 )
          v54 = v37;
        v84 = *(_DWORD *)(v46 + 16);
        ++*(_QWORD *)v46;
        v53 = v84 + 1;
        if ( 4 * v53 < 3 * v48 )
        {
          if ( v48 - *(_DWORD *)(v46 + 20) - v53 > v48 >> 3 )
            goto LABEL_44;
          v108 = v33;
          v112 = ((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4);
          sub_116E750(v46, v48);
          v85 = *(_DWORD *)(v46 + 24);
          if ( !v85 )
          {
LABEL_177:
            ++*(_DWORD *)(v46 + 16);
            BUG();
          }
          v86 = v85 - 1;
          v87 = *(_QWORD *)(v46 + 8);
          v88 = 0;
          v33 = v108;
          v89 = 1;
          v90 = v86 & v112;
          v53 = *(_DWORD *)(v46 + 16) + 1;
          v54 = (__int64 *)(v87 + 16LL * (v86 & v112));
          v91 = *v54;
          if ( v47 == *v54 )
          {
LABEL_44:
            *(_DWORD *)(v46 + 16) = v53;
            if ( *v54 != -4096 )
              --*(_DWORD *)(v46 + 20);
            *v54 = v47;
            v54[1] = 0;
            v54[1] = v45;
            a2 = *(unsigned int *)(a1 + 48);
            if ( !(_DWORD)a2 )
              goto LABEL_47;
            goto LABEL_35;
          }
          while ( v91 != -4096 )
          {
            if ( !v88 && v91 == -8192 )
              v88 = v54;
            v90 = v86 & (v89 + v90);
            v54 = (__int64 *)(v87 + 16LL * v90);
            v91 = *v54;
            if ( v47 == *v54 )
              goto LABEL_44;
            ++v89;
          }
LABEL_100:
          if ( v88 )
            v54 = v88;
          goto LABEL_44;
        }
      }
      else
      {
        ++*(_QWORD *)v46;
      }
      v109 = v33;
      sub_116E750(v46, 2 * v48);
      v49 = *(_DWORD *)(v46 + 24);
      if ( !v49 )
        goto LABEL_177;
      v50 = v49 - 1;
      v51 = *(_QWORD *)(v46 + 8);
      v33 = v109;
      v52 = (v49 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
      v53 = *(_DWORD *)(v46 + 16) + 1;
      v54 = (__int64 *)(v51 + 16LL * v52);
      v55 = *v54;
      if ( v47 == *v54 )
        goto LABEL_44;
      v107 = 1;
      v88 = 0;
      while ( v55 != -4096 )
      {
        if ( !v88 && v55 == -8192 )
          v88 = v54;
        v52 = v50 & (v107 + v52);
        v54 = (__int64 *)(v51 + 16LL * v52);
        v55 = *v54;
        if ( v47 == *v54 )
          goto LABEL_44;
        ++v107;
      }
      goto LABEL_100;
    }
LABEL_38:
    v34 += 8;
  }
  while ( v33 != v34 );
  v9 = v120;
  v6 = v117;
LABEL_56:
  if ( v133[0] != v134 )
    _libc_free(v133[0], a2);
  if ( !*(_QWORD *)(v9 + 8) )
  {
    v62 = *(_QWORD *)a1;
    v63 = *(_DWORD *)(v9 + 40);
    v121 = *(_QWORD *)v9;
    v135 = 260;
    v133[0] = (const char *)(v62 + 16);
    v64 = *(_QWORD *)(v62 + 8);
    v65 = sub_BD2DA0(80);
    v66 = v65;
    if ( v65 )
    {
      v115 = (_QWORD *)v65;
      v118 = v65;
      sub_B44260(v65, v64, 55, 0x8000000u, 0, 0);
      *(_DWORD *)(v118 + 72) = v63;
      sub_BD6B50((unsigned __int8 *)v118, v133);
      sub_BD2A10(v118, *(_DWORD *)(v118 + 72), 1);
      v67 = v115;
      v66 = v118;
    }
    else
    {
      v67 = 0;
    }
    v119 = v66;
    v68 = v116;
    LOWORD(v68) = 1;
    v116 = v68;
    sub_B44220(v67, *(_QWORD *)(v121 + 56), v68);
    a2 = v9;
    *(_QWORD *)(v9 + 8) = v119;
    *sub_11D31A0(*(_QWORD *)(a1 + 8), (__int64 *)v9) = v119;
  }
LABEL_3:
  if ( v127 != ++v6 )
    goto LABEL_4;
LABEL_13:
  result = *(unsigned int *)(v122 + 8);
  v17 = *(_QWORD *)v122 + 8 * result;
  if ( *(_QWORD *)v122 != v17 )
  {
    v18 = *(_QWORD *)v122;
    v19 = a1;
    while ( 1 )
    {
      while ( 1 )
      {
        v21 = *(_QWORD *)(v17 - 8);
        result = *(_QWORD *)(v21 + 16);
        if ( result != v21 )
          break;
        v20 = *(_QWORD *)(v21 + 8);
        if ( *(_BYTE *)v20 == 84 )
        {
          result = *(_DWORD *)(v20 + 4) & 0x7FFFFFF;
          if ( (*(_DWORD *)(v20 + 4) & 0x7FFFFFF) == 0 )
          {
            if ( *(_DWORD *)(v21 + 40) )
            {
              v128 = v18;
              v124 = v17;
              v69 = 0;
              v70 = *(_QWORD *)(v17 - 8);
              v123 = v19;
              while ( 1 )
              {
                v76 = *(__int64 **)(*(_QWORD *)(v70 + 48) + 8LL * v69);
                v77 = *v76;
                v78 = *(_QWORD *)(v76[2] + 8);
                if ( (_DWORD)result == *(_DWORD *)(v20 + 72) )
                {
                  sub_B48D90(v20);
                  LODWORD(result) = *(_DWORD *)(v20 + 4) & 0x7FFFFFF;
                }
                v71 = (result + 1) & 0x7FFFFFF;
                v72 = v71 | *(_DWORD *)(v20 + 4) & 0xF8000000;
                v73 = (_QWORD *)(*(_QWORD *)(v20 - 8) + 32LL * (unsigned int)(v71 - 1));
                *(_DWORD *)(v20 + 4) = v72;
                if ( *v73 )
                {
                  a5 = v73[2];
                  v74 = v73[1];
                  *(_QWORD *)a5 = v74;
                  if ( v74 )
                  {
                    a5 = v73[2];
                    *(_QWORD *)(v74 + 16) = a5;
                  }
                }
                *v73 = v78;
                if ( v78 )
                {
                  v75 = *(_QWORD *)(v78 + 16);
                  a5 = v78 + 16;
                  v73[1] = v75;
                  if ( v75 )
                  {
                    a6 = (__int64)(v73 + 1);
                    *(_QWORD *)(v75 + 16) = v73 + 1;
                  }
                  v73[2] = a5;
                  *(_QWORD *)(v78 + 16) = v73;
                }
                ++v69;
                result = *(_QWORD *)(v20 - 8)
                       + 32LL * *(unsigned int *)(v20 + 72)
                       + 8LL * ((*(_DWORD *)(v20 + 4) & 0x7FFFFFFu) - 1);
                *(_QWORD *)result = v77;
                if ( *(_DWORD *)(v70 + 40) == v69 )
                  break;
                LODWORD(result) = *(_DWORD *)(v20 + 4) & 0x7FFFFFF;
              }
              v18 = v128;
              v17 = v124;
              v19 = v123;
            }
            v79 = *(_QWORD *)(v19 + 16);
            if ( v79 )
            {
              result = *(unsigned int *)(v79 + 8);
              if ( result + 1 > (unsigned __int64)*(unsigned int *)(v79 + 12) )
              {
                v132 = v19;
                sub_C8D5F0(*(_QWORD *)(v19 + 16), (const void *)(v79 + 16), result + 1, 8u, a5, a6);
                result = *(unsigned int *)(v79 + 8);
                v19 = v132;
              }
              *(_QWORD *)(*(_QWORD *)v79 + 8 * result) = v20;
              ++*(_DWORD *)(v79 + 8);
            }
          }
        }
        v17 -= 8;
        if ( v18 == v17 )
          return result;
      }
      v22 = *(_QWORD *)(v19 + 8);
      v23 = *(_QWORD *)(result + 8);
      v24 = *(_DWORD *)(v22 + 24);
      if ( !v24 )
        break;
      a6 = *(_QWORD *)(v22 + 8);
      v25 = (v24 - 1) & (((unsigned int)*(_QWORD *)v21 >> 9) ^ ((unsigned int)*(_QWORD *)v21 >> 4));
      v26 = (__int64 *)(a6 + 16LL * v25);
      a5 = *v26;
      if ( *(_QWORD *)v21 != *v26 )
      {
        v129 = 1;
        v80 = 0;
        while ( a5 != -4096 )
        {
          if ( a5 == -8192 && !v80 )
            v80 = v26;
          v25 = (v24 - 1) & (v129 + v25);
          v26 = (__int64 *)(a6 + 16LL * v25);
          a5 = *v26;
          if ( *(_QWORD *)v21 == *v26 )
            goto LABEL_20;
          ++v129;
        }
        if ( !v80 )
          v80 = v26;
        v81 = *(_DWORD *)(v22 + 16);
        ++*(_QWORD *)v22;
        v82 = v81 + 1;
        if ( 4 * v82 < 3 * v24 )
        {
          if ( v24 - *(_DWORD *)(v22 + 20) - v82 <= v24 >> 3 )
          {
            v126 = v19;
            sub_116E750(v22, v24);
            v98 = *(_DWORD *)(v22 + 24);
            if ( !v98 )
            {
LABEL_176:
              ++*(_DWORD *)(v22 + 16);
              BUG();
            }
            a6 = *(_QWORD *)v21;
            v99 = *(_QWORD *)(v22 + 8);
            v97 = 0;
            v131 = v98 - 1;
            v19 = v126;
            v100 = 1;
            v101 = (v98 - 1) & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
            v82 = *(_DWORD *)(v22 + 16) + 1;
            v80 = (__int64 *)(v99 + 16LL * (v131 & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4))));
            a5 = *v80;
            if ( *(_QWORD *)v21 != *v80 )
            {
              while ( a5 != -4096 )
              {
                if ( a5 == -8192 && !v97 )
                  v97 = v80;
                v101 = v131 & (v100 + v101);
                v80 = (__int64 *)(v99 + 16LL * v101);
                a5 = *v80;
                if ( a6 == *v80 )
                  goto LABEL_88;
                ++v100;
              }
              goto LABEL_129;
            }
          }
          goto LABEL_88;
        }
LABEL_117:
        v125 = v19;
        sub_116E750(v22, 2 * v24);
        v93 = *(_DWORD *)(v22 + 24);
        if ( !v93 )
          goto LABEL_176;
        a6 = *(_QWORD *)v21;
        v94 = *(_QWORD *)(v22 + 8);
        v130 = v93 - 1;
        v19 = v125;
        v95 = (v93 - 1) & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
        v82 = *(_DWORD *)(v22 + 16) + 1;
        v80 = (__int64 *)(v94 + 16LL * (v130 & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4))));
        a5 = *v80;
        if ( *v80 != *(_QWORD *)v21 )
        {
          v96 = 1;
          v97 = 0;
          while ( a5 != -4096 )
          {
            if ( !v97 && a5 == -8192 )
              v97 = v80;
            v95 = v130 & (v96 + v95);
            v80 = (__int64 *)(v94 + 16LL * v95);
            a5 = *v80;
            if ( a6 == *v80 )
              goto LABEL_88;
            ++v96;
          }
LABEL_129:
          if ( v97 )
            v80 = v97;
        }
LABEL_88:
        *(_DWORD *)(v22 + 16) = v82;
        if ( *v80 != -4096 )
          --*(_DWORD *)(v22 + 20);
        v83 = *(_QWORD *)v21;
        v80[1] = 0;
        *v80 = v83;
        result = (__int64)(v80 + 1);
        goto LABEL_21;
      }
LABEL_20:
      result = (__int64)(v26 + 1);
LABEL_21:
      v17 -= 8;
      *(_QWORD *)result = v23;
      if ( v18 == v17 )
        return result;
    }
    ++*(_QWORD *)v22;
    goto LABEL_117;
  }
  return result;
}
