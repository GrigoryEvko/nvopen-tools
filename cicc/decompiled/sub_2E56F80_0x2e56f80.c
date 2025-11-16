// Function: sub_2E56F80
// Address: 0x2e56f80
//
__int64 __fastcall sub_2E56F80(__int64 ***a1, __int64 a2)
{
  __int64 *v5; // rsi
  __int64 v6; // rax
  int v7; // ebx
  int v8; // r14d
  char v9; // r9
  int v10; // eax
  __int64 **v11; // r12
  __int64 v12; // rbx
  __int64 **v13; // r12
  __int64 v14; // rcx
  unsigned int v15; // eax
  __int64 **v16; // r9
  int v17; // r11d
  __int64 *v18; // rcx
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 **v23; // rdx
  __int64 result; // rax
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 **v27; // r13
  __int64 v28; // rbx
  __int64 **v29; // rdx
  __int64 v30; // rcx
  unsigned int v31; // eax
  bool v32; // zf
  __int64 v33; // rax
  int v34; // esi
  int v35; // edx
  __int64 v36; // rdx
  unsigned __int64 v37; // rcx
  __int64 v38; // rax
  _QWORD *v39; // r13
  char v40; // al
  int v41; // ecx
  _DWORD *v42; // rax
  __int64 v43; // rdi
  __int64 v44; // rsi
  unsigned __int64 v45; // rax
  _QWORD *v46; // rax
  _QWORD *v47; // rcx
  __int64 v48; // rax
  unsigned __int64 v49; // rax
  _DWORD *v50; // rax
  int v51; // edx
  _DWORD *v52; // rax
  int v53; // edx
  _DWORD *v54; // rax
  int v55; // edx
  __int64 i; // rax
  unsigned __int64 v57; // rdx
  unsigned int v58; // ecx
  __int64 ***v59; // rdx
  __int64 **v60; // r11
  int v61; // esi
  __int64 v62; // rdx
  int v63; // r13d
  __int64 v64; // r11
  unsigned int v65; // ecx
  __int64 v66; // rax
  __int64 v67; // r10
  int v68; // r13d
  unsigned int v69; // ecx
  __int64 v70; // rsi
  __int64 v71; // r10
  unsigned int v72; // r12d
  __int64 v73; // r8
  __int64 v74; // r9
  __int64 *v75; // r13
  __int64 *v76; // r12
  __int64 v77; // r14
  int v78; // eax
  int v79; // edx
  __int64 v80; // rdi
  int v81; // esi
  _QWORD *v82; // rax
  int v83; // edx
  __int64 v84; // rdx
  int v85; // esi
  __int64 *v86; // r14
  __int64 *j; // rdx
  __int64 v88; // rsi
  unsigned int v89; // eax
  __int64 *v90; // rax
  unsigned int v91; // eax
  __int64 v92; // rax
  __int64 **v93; // rcx
  int v94; // edi
  int v95; // r11d
  __int64 **v96; // rcx
  __int64 **v97; // rsi
  __int64 *v98; // rdx
  __int64 **v99; // rax
  unsigned int v100; // r10d
  __int64 **v101; // r9
  unsigned int v102; // edi
  __int64 **v103; // rcx
  __int64 *v104; // rbx
  int v105; // ecx
  int v106; // r12d
  int v107; // ebx
  int v108; // edx
  int v109; // ebx
  int v110; // [rsp+10h] [rbp-F0h]
  __int64 v111; // [rsp+18h] [rbp-E8h]
  __int64 *v112; // [rsp+18h] [rbp-E8h]
  __int64 **v113; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v114; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v115; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v116; // [rsp+40h] [rbp-C0h] BYREF
  __int64 **v117; // [rsp+48h] [rbp-B8h]
  __int64 v118; // [rsp+50h] [rbp-B0h]
  unsigned int v119; // [rsp+58h] [rbp-A8h]
  __int64 v120; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v121; // [rsp+68h] [rbp-98h]
  __int64 v122; // [rsp+70h] [rbp-90h]
  unsigned int v123; // [rsp+78h] [rbp-88h]
  __int64 *v124; // [rsp+80h] [rbp-80h] BYREF
  __int64 v125; // [rsp+88h] [rbp-78h]
  _BYTE v126[112]; // [rsp+90h] [rbp-70h] BYREF

  v5 = **a1;
  v6 = sub_DF9A70(a2);
  v7 = v6;
  v124 = (__int64 *)v6;
  v8 = v6;
  v9 = sub_DF9980(a2);
  v10 = 7;
  if ( v9 )
  {
    v50 = sub_C94E20((__int64)qword_4F862F0);
    v51 = v50 ? *v50 : LODWORD(qword_4F862F0[2]);
    v10 = 7;
    if ( v51 >= 0 )
    {
      v52 = sub_C94E20((__int64)qword_4F862F0);
      v53 = v52 ? *v52 : LODWORD(qword_4F862F0[2]);
      v10 = 7;
      if ( v53 <= 10 )
      {
        v54 = sub_C94E20((__int64)qword_4F862F0);
        v55 = v54 ? *v54 : LODWORD(qword_4F862F0[2]);
        v10 = 7;
        if ( (unsigned int)(v55 + 4) <= 0x12 )
        {
          v8 = v7 + (v55 - 5) * v7 / 10;
          v10 = 7 * (v55 - 5) / 10 + 7;
        }
      }
    }
  }
  v11 = *a1;
  *((_DWORD *)a1 + 10) = v8;
  *((_DWORD *)a1 + 11) = v10;
  v12 = (__int64)v11[41];
  v13 = v11 + 40;
  if ( (__int64 **)v12 != v13 )
  {
    while ( 1 )
    {
      v23 = a1[2];
      if ( v12 )
      {
        v14 = (unsigned int)(*(_DWORD *)(v12 + 24) + 1);
        v15 = *(_DWORD *)(v12 + 24) + 1;
      }
      else
      {
        v14 = 0;
        v15 = 0;
      }
      if ( v15 >= *((_DWORD *)v23 + 8) || !v23[3][v14] )
        goto LABEL_11;
      v5 = (__int64 *)*((unsigned int *)a1 + 34);
      v120 = v12;
      if ( !(_DWORD)v5 )
        break;
      v16 = a1[15];
      v17 = 1;
      v18 = 0;
      v19 = ((_DWORD)v5 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v20 = (__int64 *)&v16[2 * v19];
      v21 = *v20;
      if ( *v20 != v12 )
      {
        while ( v21 != -4096 )
        {
          if ( v21 != -8192 || v18 )
            v20 = v18;
          v19 = ((_DWORD)v5 - 1) & (v17 + v19);
          v21 = (__int64)v16[2 * v19];
          if ( v21 == v12 )
          {
            v20 = (__int64 *)&v16[2 * v19];
            goto LABEL_9;
          }
          ++v17;
          v18 = v20;
          v20 = (__int64 *)&v16[2 * v19];
        }
        if ( !v18 )
          v18 = v20;
        v78 = *((_DWORD *)a1 + 32);
        a1[14] = (__int64 **)((char *)a1[14] + 1);
        v79 = v78 + 1;
        v124 = v18;
        if ( 4 * (v78 + 1) < (unsigned int)(3 * (_DWORD)v5) )
        {
          v80 = v12;
          if ( (int)v5 - *((_DWORD *)a1 + 33) - v79 <= (unsigned int)v5 >> 3 )
            goto LABEL_95;
LABEL_90:
          *((_DWORD *)a1 + 32) = v79;
          if ( *v18 != -4096 )
            --*((_DWORD *)a1 + 33);
          *v18 = v80;
          v22 = 0;
          v18[1] = 0;
          goto LABEL_10;
        }
LABEL_94:
        LODWORD(v5) = 2 * (_DWORD)v5;
LABEL_95:
        sub_2E56D40((__int64)(a1 + 14), (int)v5);
        v5 = &v120;
        sub_2E514F0((__int64)(a1 + 14), &v120, &v124);
        v80 = v120;
        v18 = v124;
        v79 = *((_DWORD *)a1 + 32) + 1;
        goto LABEL_90;
      }
LABEL_9:
      v22 = v20[1];
LABEL_10:
      *(_QWORD *)(v22 + 16) = a1[5];
LABEL_11:
      v12 = *(_QWORD *)(v12 + 8);
      if ( v13 == (__int64 **)v12 )
        goto LABEL_14;
    }
    a1[14] = (__int64 **)((char *)a1[14] + 1);
    v124 = 0;
    goto LABEL_94;
  }
LABEL_14:
  result = sub_DF9AE0(a2);
  if ( !(_BYTE)result )
    return result;
  result = sub_307BEA0(a1, v5);
  if ( !(_BYTE)result )
    return result;
  v27 = *a1;
  v120 = 0;
  v124 = (__int64 *)v126;
  v125 = 0x800000000LL;
  v121 = 0;
  v122 = 0;
  v123 = 0;
  v28 = (__int64)v27[41];
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v119 = 0;
  v111 = (__int64)(v27 + 40);
  if ( (__int64 **)v28 == v27 + 40 )
  {
    if ( !*((_BYTE *)a1 + 48) )
    {
      v43 = 0;
      v44 = 0;
      goto LABEL_43;
    }
    v113 = v27 + 40;
    i = 0;
    goto LABEL_67;
  }
  do
  {
    v29 = a1[2];
    if ( v28 )
    {
      v30 = (unsigned int)(*(_DWORD *)(v28 + 24) + 1);
      v31 = *(_DWORD *)(v28 + 24) + 1;
    }
    else
    {
      v30 = 0;
      v31 = 0;
    }
    if ( v31 < *((_DWORD *)v29 + 8) && v29[3][v30] )
    {
      v113 = (__int64 **)v28;
      v32 = (unsigned __int8)sub_2E50510((__int64)&v116, (__int64 *)&v113, &v114) == 0;
      v33 = v114;
      if ( !v32 )
      {
LABEL_28:
        *(_DWORD *)(v33 + 8) = -1;
        v37 = *(_QWORD *)(v28 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v37 )
LABEL_174:
          BUG();
        v38 = *(_QWORD *)v37;
        v39 = (_QWORD *)(*(_QWORD *)(v28 + 48) & 0xFFFFFFFFFFFFFFF8LL);
        if ( (*(_QWORD *)v37 & 4) == 0 && (*(_BYTE *)(v37 + 44) & 4) != 0 )
        {
          while ( 1 )
          {
            v45 = v38 & 0xFFFFFFFFFFFFFFF8LL;
            v39 = (_QWORD *)v45;
            if ( (*(_BYTE *)(v45 + 44) & 4) == 0 )
              break;
            v38 = *(_QWORD *)v45;
          }
        }
        while ( 1 )
        {
          if ( (_QWORD *)(v28 + 48) == v39 )
            goto LABEL_38;
          v115 = sub_307BA30(a1, v39);
          if ( BYTE4(v115) )
          {
            if ( (unsigned int)(v115 - 24) <= 0xE8 )
            {
              v110 = v115;
              if ( (v115 & 7) == 0 )
                break;
            }
          }
          v46 = (_QWORD *)(*v39 & 0xFFFFFFFFFFFFFFF8LL);
          v47 = v46;
          if ( !v46 )
            goto LABEL_174;
          v39 = (_QWORD *)(*v39 & 0xFFFFFFFFFFFFFFF8LL);
          v48 = *v46;
          if ( (v48 & 4) == 0 && (*((_BYTE *)v47 + 44) & 4) != 0 )
          {
            while ( 1 )
            {
              v49 = v48 & 0xFFFFFFFFFFFFFFF8LL;
              v39 = (_QWORD *)v49;
              if ( (*(_BYTE *)(v49 + 44) & 4) == 0 )
                break;
              v48 = *(_QWORD *)v49;
            }
          }
        }
        v113 = (__int64 **)v28;
        v40 = sub_2E50510((__int64)&v120, (__int64 *)&v113, &v114);
        v41 = v110;
        if ( v40 )
        {
          v42 = (_DWORD *)(v114 + 8);
LABEL_37:
          *v42 = v41;
          *((_BYTE *)a1 + 48) = 1;
          goto LABEL_38;
        }
        v81 = v123;
        v82 = (_QWORD *)v114;
        ++v120;
        v83 = v122 + 1;
        v115 = v114;
        if ( 4 * ((int)v122 + 1) >= 3 * v123 )
        {
          v81 = 2 * v123;
        }
        else
        {
          v26 = v123 >> 3;
          if ( v123 - HIDWORD(v122) - v83 > (unsigned int)v26 )
          {
LABEL_98:
            LODWORD(v122) = v83;
            if ( *v82 != -4096 )
              --HIDWORD(v122);
            v84 = (__int64)v113;
            v42 = v82 + 1;
            *v42 = 0;
            *((_QWORD *)v42 - 1) = v84;
            goto LABEL_37;
          }
        }
        sub_2E515B0((__int64)&v120, v81);
        sub_2E50510((__int64)&v120, (__int64 *)&v113, &v115);
        v41 = v110;
        v83 = v122 + 1;
        v82 = (_QWORD *)v115;
        goto LABEL_98;
      }
      v34 = v119;
      v115 = v114;
      ++v116;
      v35 = v118 + 1;
      v26 = 2 * v119;
      if ( 4 * ((int)v118 + 1) >= 3 * v119 )
      {
        v34 = 2 * v119;
      }
      else if ( v119 - HIDWORD(v118) - v35 > v119 >> 3 )
      {
LABEL_25:
        LODWORD(v118) = v35;
        if ( *(_QWORD *)v33 != -4096 )
          --HIDWORD(v118);
        v36 = (__int64)v113;
        *(_DWORD *)(v33 + 8) = 0;
        *(_QWORD *)v33 = v36;
        goto LABEL_28;
      }
      sub_2E515B0((__int64)&v116, v34);
      sub_2E50510((__int64)&v116, (__int64 *)&v113, &v115);
      v35 = v118 + 1;
      v33 = v115;
      goto LABEL_25;
    }
LABEL_38:
    v28 = *(_QWORD *)(v28 + 8);
  }
  while ( v111 != v28 );
  if ( !*((_BYTE *)a1 + 48) )
  {
    if ( v124 != (__int64 *)v126 )
      _libc_free((unsigned __int64)v124);
    v43 = v121;
    v44 = 16LL * v123;
    goto LABEL_43;
  }
  v111 = (__int64)(*a1)[41];
  i = (unsigned int)v125;
  v113 = (__int64 **)v111;
  v57 = (unsigned int)v125 + 1LL;
  if ( HIDWORD(v125) < v57 )
  {
    sub_C8D5F0((__int64)&v124, v126, v57, 8u, v25, v26);
    i = (unsigned int)v125;
  }
LABEL_67:
  v124[i] = v111;
  LODWORD(i) = v125 + 1;
  LODWORD(v125) = v125 + 1;
  if ( v123 )
  {
    v58 = (v123 - 1) & (((unsigned int)v113 >> 9) ^ ((unsigned int)v113 >> 4));
    v59 = (__int64 ***)(v121 + 16LL * v58);
    v60 = *v59;
    if ( v113 == *v59 )
    {
LABEL_69:
      if ( (__int64 ***)(v121 + 16LL * v123) != v59 )
        goto LABEL_70;
    }
    else
    {
      v108 = 1;
      while ( v60 != (__int64 **)-4096LL )
      {
        v109 = v108 + 1;
        v58 = (v123 - 1) & (v108 + v58);
        v59 = (__int64 ***)(v121 + 16LL * v58);
        v60 = *v59;
        if ( v113 == *v59 )
          goto LABEL_69;
        v108 = v109;
      }
    }
  }
  v107 = *((_DWORD *)a1 + 10);
  *(_DWORD *)sub_2E51790((__int64)&v120, (__int64 *)&v113) = v107;
  LODWORD(i) = v125;
LABEL_70:
  while ( 2 )
  {
    if ( (_DWORD)i )
    {
      v61 = v119;
      v62 = v124[(unsigned int)i - 1];
      LODWORD(v125) = i - 1;
      v114 = v62;
      if ( v119 )
      {
        v63 = 1;
        v64 = 0;
        v65 = (v119 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
        v66 = (__int64)&v117[2 * v65];
        v67 = *(_QWORD *)v66;
        if ( v62 == *(_QWORD *)v66 )
        {
LABEL_73:
          v68 = *(_DWORD *)(v66 + 8);
          goto LABEL_74;
        }
        while ( v67 != -4096 )
        {
          if ( !v64 && v67 == -8192 )
            v64 = v66;
          v65 = (v119 - 1) & (v63 + v65);
          v66 = (__int64)&v117[2 * v65];
          v67 = *(_QWORD *)v66;
          if ( v62 == *(_QWORD *)v66 )
            goto LABEL_73;
          ++v63;
        }
        if ( v64 )
          v66 = v64;
        ++v116;
        v94 = v118 + 1;
        v115 = v66;
        if ( 4 * ((int)v118 + 1) < 3 * v119 )
        {
          if ( v119 - HIDWORD(v118) - v94 > v119 >> 3 )
          {
LABEL_128:
            LODWORD(v118) = v94;
            if ( *(_QWORD *)v66 != -4096 )
              --HIDWORD(v118);
            *(_QWORD *)v66 = v62;
            v68 = 0;
            *(_DWORD *)(v66 + 8) = 0;
            v62 = v114;
LABEL_74:
            if ( v123 )
            {
              v69 = (v123 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
              v70 = v121 + 16LL * v69;
              v71 = *(_QWORD *)v70;
              if ( *(_QWORD *)v70 == v62 )
              {
LABEL_76:
                if ( v70 != v121 + 16LL * v123 )
                {
                  v72 = *(_DWORD *)(v70 + 8);
LABEL_78:
                  LODWORD(i) = v125;
                  if ( v68 != v72 )
                  {
                    *(_DWORD *)sub_2E51790((__int64)&v116, &v114) = v72;
                    v75 = *(__int64 **)(v114 + 112);
                    v76 = &v75[*(unsigned int *)(v114 + 120)];
                    for ( i = (unsigned int)v125; v76 != v75; LODWORD(v125) = v125 + 1 )
                    {
                      v77 = *v75;
                      if ( i + 1 > (unsigned __int64)HIDWORD(v125) )
                      {
                        sub_C8D5F0((__int64)&v124, v126, i + 1, 8u, v73, v74);
                        i = (unsigned int)v125;
                      }
                      ++v75;
                      v124[i] = v77;
                      i = (unsigned int)(v125 + 1);
                    }
                  }
                  continue;
                }
              }
              else
              {
                v85 = 1;
                while ( v71 != -4096 )
                {
                  v95 = v85 + 1;
                  v69 = (v123 - 1) & (v85 + v69);
                  v70 = v121 + 16LL * v69;
                  v71 = *(_QWORD *)v70;
                  if ( v62 == *(_QWORD *)v70 )
                    goto LABEL_76;
                  v85 = v95;
                }
              }
            }
            v86 = *(__int64 **)(v62 + 64);
            v72 = -1;
            for ( j = &v86[*(unsigned int *)(v62 + 72)]; j != v86; ++v86 )
            {
              v92 = *v86;
              v93 = a1[2];
              v115 = v92;
              if ( v92 )
              {
                v88 = (unsigned int)(*(_DWORD *)(v92 + 24) + 1);
                v89 = *(_DWORD *)(v92 + 24) + 1;
              }
              else
              {
                v88 = 0;
                v89 = 0;
              }
              if ( v89 < *((_DWORD *)v93 + 8) )
              {
                if ( v93[3][v88] )
                {
                  v112 = j;
                  v90 = sub_2E51790((__int64)&v116, &v115);
                  j = v112;
                  v91 = *(_DWORD *)v90;
                  if ( v72 > v91 )
                    v72 = v91;
                }
              }
            }
            goto LABEL_78;
          }
LABEL_133:
          sub_2E515B0((__int64)&v116, v61);
          sub_2E50510((__int64)&v116, &v114, &v115);
          v62 = v114;
          v94 = v118 + 1;
          v66 = v115;
          goto LABEL_128;
        }
      }
      else
      {
        ++v116;
        v115 = 0;
      }
      v61 = 2 * v119;
      goto LABEL_133;
    }
    break;
  }
  if ( (_DWORD)v118 )
  {
    v96 = v117;
    v97 = &v117[2 * v119];
    if ( v117 != v97 )
    {
      while ( 1 )
      {
        v98 = *v96;
        v99 = v96;
        if ( *v96 != (__int64 *)-8192LL && v98 != (__int64 *)-4096LL )
          break;
        v96 += 2;
        if ( v97 == v96 )
          goto LABEL_114;
      }
      while ( v99 != v97 )
      {
        v100 = *((_DWORD *)a1 + 34);
        v101 = a1[15];
        if ( v100 )
        {
          v102 = (v100 - 1) & (((unsigned int)v98 >> 9) ^ ((unsigned int)v98 >> 4));
          v103 = &v101[2 * v102];
          v104 = *v103;
          if ( v98 == *v103 )
          {
LABEL_147:
            if ( v103 != &v101[2 * v100] )
              *((_DWORD *)v103[1] + 4) = *((_DWORD *)v99 + 2);
          }
          else
          {
            v105 = 1;
            while ( v104 != (__int64 *)-4096LL )
            {
              v106 = v105 + 1;
              v102 = (v100 - 1) & (v105 + v102);
              v103 = &v101[2 * v102];
              v104 = *v103;
              if ( v98 == *v103 )
                goto LABEL_147;
              v105 = v106;
            }
          }
        }
        v99 += 2;
        if ( v99 == v97 )
          break;
        while ( 1 )
        {
          v98 = *v99;
          if ( *v99 != (__int64 *)-8192LL && v98 != (__int64 *)-4096LL )
            break;
          v99 += 2;
          if ( v97 == v99 )
            goto LABEL_114;
        }
      }
    }
  }
LABEL_114:
  if ( v124 != (__int64 *)v126 )
    _libc_free((unsigned __int64)v124);
  v43 = v121;
  v44 = 16LL * v123;
LABEL_43:
  sub_C7D6A0(v43, v44, 8);
  return sub_C7D6A0((__int64)v117, 16LL * v119, 8);
}
