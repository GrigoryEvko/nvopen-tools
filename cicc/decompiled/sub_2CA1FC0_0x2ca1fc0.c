// Function: sub_2CA1FC0
// Address: 0x2ca1fc0
//
void __fastcall sub_2CA1FC0(
        __int64 a1,
        __int64 ******a2,
        __int64 a3,
        __int64 a4,
        __int64 **a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v9; // r14
  __int64 ****v10; // rbx
  __int64 *v11; // rsi
  __int64 *v12; // rcx
  __int64 v13; // r12
  __int64 v14; // rsi
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  int v17; // eax
  __int64 v18; // rdx
  bool v19; // cf
  _QWORD *v20; // r13
  unsigned int *v21; // rax
  __int64 **v22; // rbx
  __int64 **v23; // rax
  __int64 **v24; // r12
  __int64 *v25; // r14
  _QWORD *v26; // rsi
  _QWORD *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r15
  __int64 **v30; // rcx
  __int64 ****v31; // rbx
  __int64 ***v32; // rax
  __int64 **v33; // r12
  __int64 v34; // r8
  unsigned int v35; // edi
  __int64 *v36; // rcx
  __int64 v37; // rdx
  __int64 *v38; // rbx
  _QWORD *v39; // r14
  unsigned int v40; // esi
  __int64 v41; // r13
  int v42; // r8d
  int v43; // r8d
  __int64 v44; // rsi
  unsigned int v45; // eax
  __int64 *v46; // r10
  __int64 v47; // rcx
  int v48; // edx
  __int64 *v49; // rdi
  __int64 v50; // rax
  _QWORD *v51; // r13
  __int64 v52; // rax
  __int64 v53; // rsi
  __int64 *v54; // rax
  __int64 *v55; // rdi
  __int64 v56; // rcx
  unsigned __int64 v57; // rax
  int v58; // edx
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  unsigned int v62; // esi
  __int64 ***v63; // rcx
  __int64 v64; // r8
  __int64 ****v65; // r11
  int v66; // r12d
  unsigned int v67; // edx
  _QWORD *v68; // rax
  __int64 ***v69; // rdi
  _QWORD *v70; // rax
  __int64 v71; // rdi
  int v72; // eax
  int v73; // edx
  unsigned int v74; // eax
  __int64 *******v75; // rcx
  __int64 ******v76; // rsi
  int v77; // eax
  __int64 ***v78; // rax
  int v79; // r11d
  int v80; // ecx
  int v81; // ecx
  __int64 v82; // rsi
  int v83; // ecx
  __int64 *v84; // r9
  int v85; // r11d
  int v86; // r8d
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 *v89; // r8
  __int64 *v90; // rax
  int v91; // ecx
  __int64 ***v92; // rsi
  __int64 v93; // r8
  int v94; // ecx
  unsigned int v95; // edx
  __int64 ***v96; // rdi
  int v97; // r12d
  __int64 ****v98; // r10
  int v99; // ecx
  __int64 ***v100; // rsi
  __int64 v101; // r8
  int v102; // ecx
  int v103; // r12d
  unsigned int v104; // edx
  __int64 ***v105; // rdi
  _QWORD *v106; // rax
  __int64 *v107; // r8
  int v108; // r11d
  __int64 *v109; // rdi
  int v110; // ecx
  int v111; // r8d
  int v112; // edi
  unsigned int v113; // r11d
  __int64 *v114; // [rsp+8h] [rbp-120h]
  unsigned int v115; // [rsp+10h] [rbp-118h]
  __int64 v117; // [rsp+20h] [rbp-108h]
  __int64 *v118; // [rsp+38h] [rbp-F0h]
  __int64 *v119; // [rsp+48h] [rbp-E0h]
  __int64 v121; // [rsp+58h] [rbp-D0h]
  __int64 v122; // [rsp+60h] [rbp-C8h]
  __int64 *v123; // [rsp+68h] [rbp-C0h]
  unsigned int v125; // [rsp+7Ch] [rbp-ACh]
  __int64 v126; // [rsp+80h] [rbp-A8h]
  __int64 v127; // [rsp+88h] [rbp-A0h]
  __int64 *v130; // [rsp+A0h] [rbp-88h]
  __int64 v131; // [rsp+A8h] [rbp-80h]
  _QWORD *v133; // [rsp+B0h] [rbp-78h]
  __int64 v134; // [rsp+C0h] [rbp-68h] BYREF
  __int64 v135; // [rsp+C8h] [rbp-60h] BYREF
  _QWORD *v136; // [rsp+D0h] [rbp-58h] BYREF
  __int64 **v137; // [rsp+D8h] [rbp-50h] BYREF
  __int64 v138; // [rsp+E0h] [rbp-48h]
  __int64 *v139; // [rsp+E8h] [rbp-40h] BYREF
  __int64 v140; // [rsp+F0h] [rbp-38h]

  v9 = a8;
  v10 = **a2;
  sub_2C9EEF0(a1, (unsigned int *)*v10, v10[1], &v134, &v135, *(_QWORD *)(a1 + 200), 0);
  v11 = a5[1];
  v12 = *a5;
  v118 = v11;
  v119 = *a5;
  v13 = (***v10)[1];
  if ( *(_WORD *)(v13 + 24) != 5 )
    v13 = 0;
  if ( v12 == v11 )
  {
    v136 = sub_F8DB90(a6, **(_QWORD **)(v13 + 32), 0, **(_QWORD **)a4 + 24LL, 0);
    v14 = (__int64)v136;
    v107 = a5[1];
    if ( v107 == a5[2] )
    {
      sub_9281F0((__int64)a5, a5[1], &v136);
      v14 = (__int64)v136;
    }
    else
    {
      if ( v107 )
      {
        *v107 = (__int64)v136;
        v107 = a5[1];
      }
      a5[1] = v107 + 1;
    }
  }
  else
  {
    v14 = *v119;
    v136 = (_QWORD *)*v119;
  }
  v123 = sub_DA3860(*(_QWORD **)(a1 + 184), v14);
  v121 = *(_QWORD *)(*(_QWORD *)(v13 + 32) + 8LL);
  v130 = sub_DC7ED0(*(__int64 **)(a1 + 184), (__int64)v123, v121, 0, 0);
  v15 = v134;
  if ( *(_BYTE *)v134 == 23 )
  {
    if ( v134 == *(_QWORD *)((***v10)[2] + 40) )
    {
      v15 = (***v10)[2];
    }
    else
    {
      v16 = *(_QWORD *)(v134 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v16 == v134 + 48 )
      {
        v15 = 0;
      }
      else
      {
        if ( !v16 )
          BUG();
        v17 = *(unsigned __int8 *)(v16 - 24);
        v18 = v16 - 24;
        v19 = (unsigned int)(v17 - 30) < 0xB;
        v15 = 0;
        if ( v19 )
          v15 = v18;
      }
    }
  }
  v20 = sub_F8DB90(a6, (__int64)v130, 0, v15 + 24, 0);
  *sub_2C92EC0(a7, (__int64 *)**a2) = (__int64)v20;
  v21 = (unsigned int *)*v10;
  v22 = **v10;
  v23 = &v22[v21[2]];
  if ( v23 != v22 )
  {
    v24 = v23;
    do
    {
      v25 = *v22;
      v26 = v20;
      if ( !sub_D968A0(**v22) )
      {
        v27 = sub_DC7ED0(*(__int64 **)(a1 + 184), (__int64)v130, *v25, 0, 0);
        v26 = sub_F8DB90(a6, (__int64)v27, 0, v25[2] + 24, 0);
      }
      ++v22;
      sub_2C95850(v25[3], (__int64)v26, v25[2]);
    }
    while ( v22 != v24 );
    v9 = a8;
  }
  if ( v119 == v118 )
  {
    v106 = sub_F8DB90(a6, a3, 0, *(_QWORD *)(*(_QWORD *)a4 + 8LL) + 24LL, 0);
    v114 = sub_DA3860(*(_QWORD **)(a1 + 184), (__int64)v106);
  }
  else
  {
    v114 = 0;
  }
  v28 = 0;
  v29 = v9;
  v125 = 1;
  v30 = (__int64 **)*a2;
  if ( *a2 != a2[1] )
  {
    while ( 1 )
    {
      v122 = v28;
      sub_2C9EEF0(a1, (unsigned int *)*v30[v28], (_QWORD **)v30[v28][1], &v134, &v135, *(_QWORD *)(a1 + 200), 0);
      if ( v119 == v118 )
      {
        v90 = sub_DC7ED0(*(__int64 **)(a1 + 184), (__int64)v123, (__int64)v114, 0, 0);
        v136 = 0;
        v123 = v90;
      }
      else
      {
        v136 = (_QWORD *)(*a5)[v125];
      }
      v31 = (*a2)[v122];
      v32 = v31[1];
      v33 = *v32;
      v131 = (__int64)&(*v32)[*((unsigned int *)v32 + 2)];
      if ( (__int64 **)v131 != *v32 )
        break;
      v61 = a7;
      v133 = 0;
      v62 = *(_DWORD *)(a7 + 24);
      if ( !v62 )
        goto LABEL_90;
LABEL_49:
      v63 = v31[1];
      v64 = *(_QWORD *)(v61 + 8);
      v65 = 0;
      v66 = 1;
      v67 = (v62 - 1) & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
      v68 = (_QWORD *)(v64 + 16LL * v67);
      v69 = (__int64 ***)*v68;
      if ( (__int64 ***)*v68 == v63 )
      {
LABEL_50:
        v70 = v68 + 1;
      }
      else
      {
        while ( v69 != (__int64 ***)-4096LL )
        {
          if ( v69 == (__int64 ***)-8192LL && !v65 )
            v65 = (__int64 ****)v68;
          v67 = (v62 - 1) & (v66 + v67);
          v68 = (_QWORD *)(v64 + 16LL * v67);
          v69 = (__int64 ***)*v68;
          if ( v63 == (__int64 ***)*v68 )
            goto LABEL_50;
          ++v66;
        }
        if ( !v65 )
          v65 = (__int64 ****)v68;
        ++*(_QWORD *)a7;
        v77 = *(_DWORD *)(a7 + 16) + 1;
        if ( 4 * v77 >= 3 * v62 )
          goto LABEL_91;
        if ( v62 - *(_DWORD *)(a7 + 20) - v77 > v62 >> 3 )
          goto LABEL_68;
        sub_2C92CE0(a7, v62);
        v99 = *(_DWORD *)(a7 + 24);
        if ( !v99 )
          goto LABEL_146;
        v100 = v31[1];
        v101 = *(_QWORD *)(a7 + 8);
        v102 = v99 - 1;
        v98 = 0;
        v103 = 1;
        v104 = v102 & (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4));
        v65 = (__int64 ****)(v101 + 16LL * v104);
        v77 = *(_DWORD *)(a7 + 16) + 1;
        v105 = *v65;
        if ( *v65 != v100 )
        {
          while ( v105 != (__int64 ***)-4096LL )
          {
            if ( !v98 && v105 == (__int64 ***)-8192LL )
              v98 = v65;
            v104 = v102 & (v103 + v104);
            v65 = (__int64 ****)(v101 + 16LL * v104);
            v105 = *v65;
            if ( v100 == *v65 )
              goto LABEL_68;
            ++v103;
          }
          goto LABEL_95;
        }
LABEL_68:
        *(_DWORD *)(a7 + 16) = v77;
        if ( *v65 != (__int64 ***)-4096LL )
          --*(_DWORD *)(a7 + 20);
        v78 = v31[1];
        v65[1] = 0;
        *v65 = v78;
        v70 = v65 + 1;
      }
      v28 = v125;
      *v70 = v133;
      ++v125;
      v30 = (__int64 **)*a2;
      if ( v28 == a2[1] - *a2 )
        goto LABEL_52;
    }
    v133 = 0;
    while ( 1 )
    {
      v38 = *v33;
      if ( !v133 )
      {
        v53 = (__int64)v136;
        if ( !v136 )
        {
          v88 = v117;
          LOWORD(v88) = 0;
          v117 = v88;
          v136 = sub_F8DB90(a6, (__int64)v123, 0, *(_QWORD *)(*(_QWORD *)a4 + 8LL * v125) + 24LL, 0);
          v53 = (__int64)v136;
          v89 = a5[1];
          if ( v89 == a5[2] )
          {
            sub_9281F0((__int64)a5, a5[1], &v136);
            v53 = (__int64)v136;
          }
          else
          {
            if ( v89 )
            {
              *v89 = (__int64)v136;
              v89 = a5[1];
            }
            a5[1] = v89 + 1;
          }
        }
        v54 = sub_DA3860(*(_QWORD **)(a1 + 184), v53);
        v55 = *(__int64 **)(a1 + 184);
        v123 = v54;
        v139 = v54;
        v137 = &v139;
        v140 = v121;
        v138 = 0x200000002LL;
        v130 = sub_DC7EB0(v55, (__int64)&v137, 0, 0);
        if ( v137 != &v139 )
          _libc_free((unsigned __int64)v137);
        v56 = v135;
        if ( *(_BYTE *)v135 == 23 )
        {
          if ( v135 == *(_QWORD *)(v38[2] + 40) )
          {
            v56 = v38[2];
          }
          else
          {
            v57 = *(_QWORD *)(v135 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v57 == v135 + 48 )
            {
              v56 = 0;
            }
            else
            {
              if ( !v57 )
                BUG();
              v58 = *(unsigned __int8 *)(v57 - 24);
              v56 = 0;
              v59 = v57 - 24;
              if ( (unsigned int)(v58 - 30) < 0xB )
                v56 = v59;
            }
          }
        }
        v60 = v126;
        LOWORD(v60) = 0;
        v126 = v60;
        v133 = sub_F8DB90(a6, (__int64)v130, 0, v56 + 24, 0);
      }
      v39 = v133;
      if ( !sub_D968A0(*v38) )
      {
        v49 = *(__int64 **)(a1 + 184);
        v50 = *v38;
        v137 = &v139;
        v139 = v130;
        v140 = v50;
        v138 = 0x200000002LL;
        v51 = sub_DC7EB0(v49, (__int64)&v137, 0, 0);
        if ( v137 != &v139 )
          _libc_free((unsigned __int64)v137);
        v52 = v127;
        LOWORD(v52) = 0;
        v127 = v52;
        v39 = sub_F8DB90(a6, (__int64)v51, 0, v38[2] + 24, 0);
      }
      v40 = *(_DWORD *)(v29 + 24);
      v41 = v38[2];
      if ( !v40 )
        break;
      v34 = *(_QWORD *)(v29 + 8);
      v35 = (v40 - 1) & (((unsigned int)v41 >> 4) ^ ((unsigned int)v41 >> 9));
      v36 = (__int64 *)(v34 + 8LL * v35);
      v37 = *v36;
      if ( v41 != *v36 )
      {
        v79 = 1;
        v46 = 0;
        while ( v37 != -4096 )
        {
          if ( v46 || v37 != -8192 )
            v36 = v46;
          v35 = (v40 - 1) & (v79 + v35);
          v37 = *(_QWORD *)(v34 + 8LL * v35);
          if ( v41 == v37 )
            goto LABEL_25;
          ++v79;
          v46 = v36;
          v36 = (__int64 *)(v34 + 8LL * v35);
        }
        if ( !v46 )
          v46 = v36;
        v80 = *(_DWORD *)(v29 + 16);
        ++*(_QWORD *)v29;
        v48 = v80 + 1;
        if ( 4 * (v80 + 1) < 3 * v40 )
        {
          if ( v40 - *(_DWORD *)(v29 + 20) - v48 <= v40 >> 3 )
          {
            v115 = ((unsigned int)v41 >> 4) ^ ((unsigned int)v41 >> 9);
            sub_CF4090(v29, v40);
            v81 = *(_DWORD *)(v29 + 24);
            if ( !v81 )
            {
LABEL_148:
              ++*(_DWORD *)(v29 + 16);
              BUG();
            }
            v82 = *(_QWORD *)(v29 + 8);
            v83 = v81 - 1;
            v84 = 0;
            v85 = 1;
            v46 = (__int64 *)(v82 + 8LL * (v83 & v115));
            v86 = v83 & v115;
            v48 = *(_DWORD *)(v29 + 16) + 1;
            v87 = *v46;
            if ( v41 != *v46 )
            {
              while ( v87 != -4096 )
              {
                if ( !v84 && v87 == -8192 )
                  v84 = v46;
                v112 = v85 + 1;
                v113 = v83 & (v86 + v85);
                v46 = (__int64 *)(v82 + 8LL * v113);
                v86 = v113;
                v87 = *v46;
                if ( v41 == *v46 )
                  goto LABEL_32;
                v85 = v112;
              }
              if ( v84 )
                v46 = v84;
            }
          }
          goto LABEL_32;
        }
LABEL_30:
        sub_CF4090(v29, 2 * v40);
        v42 = *(_DWORD *)(v29 + 24);
        if ( !v42 )
          goto LABEL_148;
        v43 = v42 - 1;
        v44 = *(_QWORD *)(v29 + 8);
        v45 = v43 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
        v46 = (__int64 *)(v44 + 8LL * v45);
        v47 = *v46;
        v48 = *(_DWORD *)(v29 + 16) + 1;
        if ( v41 != *v46 )
        {
          v108 = 1;
          v109 = 0;
          while ( v47 != -4096 )
          {
            if ( v47 != -8192 || v109 )
              v46 = v109;
            v45 = v43 & (v108 + v45);
            v47 = *(_QWORD *)(v44 + 8LL * v45);
            if ( v41 == v47 )
            {
              v46 = (__int64 *)(v44 + 8LL * v45);
              goto LABEL_32;
            }
            ++v108;
            v109 = v46;
            v46 = (__int64 *)(v44 + 8LL * v45);
          }
          if ( v109 )
            v46 = v109;
        }
LABEL_32:
        *(_DWORD *)(v29 + 16) = v48;
        if ( *v46 != -4096 )
          --*(_DWORD *)(v29 + 20);
        *v46 = v41;
        v37 = v38[2];
      }
LABEL_25:
      ++v33;
      sub_2C95850(v38[3], (__int64)v39, v37);
      if ( v33 == (__int64 **)v131 )
      {
        v31 = (*a2)[v122];
        v61 = a7;
        v62 = *(_DWORD *)(a7 + 24);
        if ( v62 )
          goto LABEL_49;
LABEL_90:
        ++*(_QWORD *)a7;
LABEL_91:
        sub_2C92CE0(a7, 2 * v62);
        v91 = *(_DWORD *)(a7 + 24);
        if ( v91 )
        {
          v92 = v31[1];
          v93 = *(_QWORD *)(a7 + 8);
          v94 = v91 - 1;
          v95 = v94 & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
          v65 = (__int64 ****)(v93 + 16LL * v95);
          v77 = *(_DWORD *)(a7 + 16) + 1;
          v96 = *v65;
          if ( v92 != *v65 )
          {
            v97 = 1;
            v98 = 0;
            while ( v96 != (__int64 ***)-4096LL )
            {
              if ( !v98 && v96 == (__int64 ***)-8192LL )
                v98 = v65;
              v95 = v94 & (v97 + v95);
              v65 = (__int64 ****)(v93 + 16LL * v95);
              v96 = *v65;
              if ( v92 == *v65 )
                goto LABEL_68;
              ++v97;
            }
LABEL_95:
            if ( v98 )
              v65 = v98;
          }
          goto LABEL_68;
        }
LABEL_146:
        ++*(_DWORD *)(a7 + 16);
        BUG();
      }
    }
    ++*(_QWORD *)v29;
    goto LABEL_30;
  }
LABEL_52:
  v71 = *(_QWORD *)(a1 + 240);
  v72 = *(_DWORD *)(a1 + 256);
  if ( v72 )
  {
    v73 = v72 - 1;
    v74 = (v72 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v75 = (__int64 *******)(v71 + 8LL * v74);
    v76 = *v75;
    if ( *v75 == a2 )
    {
LABEL_54:
      *v75 = (__int64 ******)-8192LL;
      --*(_DWORD *)(a1 + 248);
      ++*(_DWORD *)(a1 + 252);
    }
    else
    {
      v110 = 1;
      while ( v76 != (__int64 ******)-4096LL )
      {
        v111 = v110 + 1;
        v74 = v73 & (v110 + v74);
        v75 = (__int64 *******)(v71 + 8LL * v74);
        v76 = *v75;
        if ( *v75 == a2 )
          goto LABEL_54;
        v110 = v111;
      }
    }
  }
  if ( *a2 )
    j_j___libc_free_0((unsigned __int64)*a2);
  j_j___libc_free_0((unsigned __int64)a2);
}
