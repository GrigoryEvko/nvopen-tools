// Function: sub_1ABE760
// Address: 0x1abe760
//
__int64 __fastcall sub_1ABE760(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 *v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r12
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // r13
  unsigned int v12; // r14d
  __int64 v13; // rax
  int v14; // edx
  int v15; // ecx
  __int64 v16; // r8
  unsigned int v17; // edx
  __int64 v18; // rdi
  __int64 v19; // r13
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // r13
  unsigned int v22; // r14d
  __int64 v23; // rax
  int v24; // edx
  int v25; // ecx
  __int64 v26; // r8
  unsigned int v27; // edx
  __int64 v28; // rdi
  __int64 v29; // r13
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // r13
  unsigned int v32; // r14d
  __int64 v33; // rax
  int v34; // edx
  int v35; // ecx
  __int64 v36; // r8
  unsigned int v37; // edx
  __int64 v38; // rdi
  __int64 v39; // r13
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // r13
  unsigned int v42; // r14d
  __int64 v43; // rax
  int v44; // edx
  int v45; // ecx
  __int64 v46; // r8
  unsigned int v47; // edx
  __int64 v48; // rdi
  int v49; // r9d
  __int64 result; // rax
  __int64 v51; // r12
  int v52; // ecx
  __int64 v53; // rdi
  unsigned int v54; // edx
  __int64 v55; // rsi
  int v56; // edx
  __int64 v57; // rax
  __int64 v58; // r13
  char *v59; // rax
  char *v60; // rdx
  char *v61; // r14
  int v62; // r9d
  int v63; // r9d
  int v64; // r9d
  __int64 v65; // r14
  int v66; // eax
  __int64 v67; // rcx
  int v68; // edx
  __int64 v69; // rdi
  int v70; // r9d
  unsigned int v71; // eax
  __int64 v72; // rsi
  _QWORD *v73; // r15
  char *v74; // rdx
  int v75; // eax
  __int64 v76; // rcx
  int v77; // edx
  __int64 v78; // rdi
  unsigned int v79; // eax
  __int64 v80; // rsi
  int v81; // r8d
  int v82; // r8d
  __int64 v83; // r14
  unsigned __int64 v84; // rdi
  __int64 v85; // r14
  unsigned __int64 v86; // rdi
  __int64 v87; // r13
  unsigned __int64 v88; // rdi
  unsigned __int64 v89; // r13
  unsigned int v90; // r14d
  __int64 v91; // rax
  int v92; // edx
  int v93; // ecx
  __int64 v94; // rdi
  unsigned int v95; // edx
  __int64 v96; // rsi
  unsigned __int64 v97; // r13
  unsigned int v98; // r14d
  __int64 v99; // rax
  int v100; // edx
  int v101; // ecx
  __int64 v102; // rdi
  unsigned int v103; // edx
  __int64 v104; // rsi
  unsigned __int64 v105; // r13
  unsigned int i; // r14d
  __int64 v107; // rax
  int v108; // edx
  int v109; // esi
  __int64 v110; // rdi
  unsigned int v111; // edx
  __int64 v112; // rcx
  int v113; // r8d
  int v114; // r8d
  int v115; // r8d
  __int64 v117; // [rsp+8h] [rbp-A8h]
  _QWORD *v118; // [rsp+10h] [rbp-A0h]
  __int64 v121; // [rsp+28h] [rbp-88h]
  char *v122; // [rsp+38h] [rbp-78h]
  __int64 v123; // [rsp+40h] [rbp-70h]
  char *v124; // [rsp+40h] [rbp-70h]
  __int64 *v125; // [rsp+48h] [rbp-68h]
  __int64 *v126; // [rsp+50h] [rbp-60h]
  int v127; // [rsp+58h] [rbp-58h]
  int v128; // [rsp+58h] [rbp-58h]
  int v129; // [rsp+58h] [rbp-58h]
  int v130; // [rsp+58h] [rbp-58h]
  __int64 v131; // [rsp+58h] [rbp-58h]
  int v132; // [rsp+58h] [rbp-58h]
  int v133; // [rsp+58h] [rbp-58h]
  int v134; // [rsp+58h] [rbp-58h]
  char v135; // [rsp+66h] [rbp-4Ah] BYREF
  char v136; // [rsp+67h] [rbp-49h] BYREF
  char *v137; // [rsp+68h] [rbp-48h] BYREF
  _QWORD v138[8]; // [rsp+70h] [rbp-40h] BYREF

  v5 = *(__int64 **)(a1 + 72);
  v6 = *v5;
  v123 = *(_QWORD *)(*v5 + 56);
  v125 = *(__int64 **)(a1 + 80);
  v7 = ((char *)v125 - (char *)v5) >> 5;
  v8 = v125 - v5;
  if ( v7 > 0 )
  {
    v9 = 0;
    v126 = &v5[4 * v7];
    while ( 1 )
    {
      v10 = sub_157EBA0(v6);
      if ( v10 )
      {
        v127 = sub_15F4D60(v10);
        v11 = sub_157EBA0(v6);
        if ( v127 )
        {
          v12 = 0;
          while ( 1 )
          {
            v13 = sub_15F4DF0(v11, v12);
            v14 = *(_DWORD *)(a1 + 64);
            if ( !v14 )
              goto LABEL_31;
            v15 = v14 - 1;
            v16 = *(_QWORD *)(a1 + 48);
            v17 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v18 = *(_QWORD *)(v16 + 8LL * v17);
            if ( v13 != v18 )
              break;
LABEL_8:
            if ( v127 == ++v12 )
              goto LABEL_9;
          }
          v49 = 1;
          while ( v18 != -8 )
          {
            v17 = v15 & (v49 + v17);
            v18 = *(_QWORD *)(v16 + 8LL * v17);
            if ( v13 == v18 )
              goto LABEL_8;
            ++v49;
          }
LABEL_31:
          if ( v9 )
          {
            if ( v9 != v13 )
              goto LABEL_33;
          }
          else
          {
            v9 = v13;
          }
          goto LABEL_8;
        }
      }
LABEL_9:
      v19 = v5[1];
      v20 = sub_157EBA0(v19);
      if ( v20 )
      {
        v128 = sub_15F4D60(v20);
        v21 = sub_157EBA0(v19);
        if ( v128 )
        {
          v22 = 0;
          while ( 1 )
          {
            v23 = sub_15F4DF0(v21, v22);
            v24 = *(_DWORD *)(a1 + 64);
            if ( !v24 )
              goto LABEL_54;
            v25 = v24 - 1;
            v26 = *(_QWORD *)(a1 + 48);
            v27 = (v24 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
            v28 = *(_QWORD *)(v26 + 8LL * v27);
            if ( v23 != v28 )
              break;
LABEL_14:
            if ( v128 == ++v22 )
              goto LABEL_15;
          }
          v62 = 1;
          while ( v28 != -8 )
          {
            v27 = v25 & (v62 + v27);
            v28 = *(_QWORD *)(v26 + 8LL * v27);
            if ( v23 == v28 )
              goto LABEL_14;
            ++v62;
          }
LABEL_54:
          if ( v9 )
          {
            if ( v23 != v9 )
            {
              ++v5;
              goto LABEL_33;
            }
          }
          else
          {
            v9 = v23;
          }
          goto LABEL_14;
        }
      }
LABEL_15:
      v29 = v5[2];
      v30 = sub_157EBA0(v29);
      if ( v30 )
      {
        v129 = sub_15F4D60(v30);
        v31 = sub_157EBA0(v29);
        if ( v129 )
        {
          v32 = 0;
          while ( 1 )
          {
            v33 = sub_15F4DF0(v31, v32);
            v34 = *(_DWORD *)(a1 + 64);
            if ( !v34 )
              goto LABEL_59;
            v35 = v34 - 1;
            v36 = *(_QWORD *)(a1 + 48);
            v37 = (v34 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
            v38 = *(_QWORD *)(v36 + 8LL * v37);
            if ( v33 != v38 )
              break;
LABEL_20:
            if ( v129 == ++v32 )
              goto LABEL_21;
          }
          v63 = 1;
          while ( v38 != -8 )
          {
            v37 = v35 & (v63 + v37);
            v38 = *(_QWORD *)(v36 + 8LL * v37);
            if ( v33 == v38 )
              goto LABEL_20;
            ++v63;
          }
LABEL_59:
          if ( v9 )
          {
            if ( v33 != v9 )
            {
              v5 += 2;
              goto LABEL_33;
            }
          }
          else
          {
            v9 = v33;
          }
          goto LABEL_20;
        }
      }
LABEL_21:
      v39 = v5[3];
      v40 = sub_157EBA0(v39);
      if ( v40 )
      {
        v130 = sub_15F4D60(v40);
        v41 = sub_157EBA0(v39);
        if ( v130 )
          break;
      }
LABEL_27:
      v5 += 4;
      if ( v126 == v5 )
      {
        v8 = v125 - v5;
        goto LABEL_109;
      }
      v6 = *v5;
    }
    v42 = 0;
    while ( 1 )
    {
      v43 = sub_15F4DF0(v41, v42);
      v44 = *(_DWORD *)(a1 + 64);
      if ( v44 )
      {
        v45 = v44 - 1;
        v46 = *(_QWORD *)(a1 + 48);
        v47 = (v44 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
        v48 = *(_QWORD *)(v46 + 8LL * v47);
        if ( v43 == v48 )
          goto LABEL_26;
        v64 = 1;
        while ( v48 != -8 )
        {
          v47 = v45 & (v64 + v47);
          v48 = *(_QWORD *)(v46 + 8LL * v47);
          if ( v43 == v48 )
            goto LABEL_26;
          ++v64;
        }
      }
      if ( v9 )
      {
        if ( v9 != v43 )
        {
          v5 += 3;
          goto LABEL_33;
        }
      }
      else
      {
        v9 = v43;
      }
LABEL_26:
      if ( ++v42 == v130 )
        goto LABEL_27;
    }
  }
  v9 = 0;
LABEL_109:
  switch ( v8 )
  {
    case 2LL:
LABEL_115:
      v85 = *v5;
      v86 = sub_157EBA0(*v5);
      if ( !v86 || (v133 = sub_15F4D60(v86), v97 = sub_157EBA0(v85), !v133) )
      {
LABEL_116:
        ++v5;
LABEL_117:
        v87 = *v5;
        v88 = sub_157EBA0(*v5);
        if ( !v88 )
          break;
        v132 = sub_15F4D60(v88);
        v89 = sub_157EBA0(v87);
        if ( !v132 )
          break;
        v90 = 0;
        while ( 1 )
        {
          v91 = sub_15F4DF0(v89, v90);
          v92 = *(_DWORD *)(a1 + 64);
          if ( v92 )
          {
            v93 = v92 - 1;
            v94 = *(_QWORD *)(a1 + 48);
            v95 = (v92 - 1) & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
            v96 = *(_QWORD *)(v94 + 8LL * v95);
            if ( v91 == v96 )
              goto LABEL_122;
            v114 = 1;
            while ( v96 != -8 )
            {
              v95 = v93 & (v114 + v95);
              v96 = *(_QWORD *)(v94 + 8LL * v95);
              if ( v91 == v96 )
                goto LABEL_122;
              ++v114;
            }
          }
          if ( v9 )
          {
            if ( v91 != v9 )
              goto LABEL_33;
          }
          else
          {
            v9 = v91;
          }
LABEL_122:
          if ( v132 == ++v90 )
            goto LABEL_112;
        }
      }
      v98 = 0;
      while ( 1 )
      {
        v99 = sub_15F4DF0(v97, v98);
        v100 = *(_DWORD *)(a1 + 64);
        if ( v100 )
        {
          v101 = v100 - 1;
          v102 = *(_QWORD *)(a1 + 48);
          v103 = (v100 - 1) & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
          v104 = *(_QWORD *)(v102 + 8LL * v103);
          if ( v104 == v99 )
            goto LABEL_128;
          v113 = 1;
          while ( v104 != -8 )
          {
            v103 = v101 & (v113 + v103);
            v104 = *(_QWORD *)(v102 + 8LL * v103);
            if ( v99 == v104 )
              goto LABEL_128;
            ++v113;
          }
        }
        if ( v9 )
        {
          if ( v99 != v9 )
            goto LABEL_33;
        }
        else
        {
          v9 = v99;
        }
LABEL_128:
        if ( v133 == ++v98 )
          goto LABEL_116;
      }
    case 3LL:
      v83 = *v5;
      v84 = sub_157EBA0(*v5);
      if ( v84 )
      {
        v134 = sub_15F4D60(v84);
        v105 = sub_157EBA0(v83);
        if ( v134 )
        {
          for ( i = 0; v134 != i; ++i )
          {
            v107 = sub_15F4DF0(v105, i);
            v108 = *(_DWORD *)(a1 + 64);
            if ( v108 )
            {
              v109 = v108 - 1;
              v110 = *(_QWORD *)(a1 + 48);
              v111 = (v108 - 1) & (((unsigned int)v107 >> 9) ^ ((unsigned int)v107 >> 4));
              v112 = *(_QWORD *)(v110 + 8LL * v111);
              if ( v107 == v112 )
                continue;
              v115 = 1;
              while ( v112 != -8 )
              {
                v111 = v109 & (v115 + v111);
                v112 = *(_QWORD *)(v110 + 8LL * v111);
                if ( v107 == v112 )
                  goto LABEL_134;
                ++v115;
              }
            }
            if ( v9 )
            {
              if ( v107 != v9 )
                goto LABEL_33;
            }
            else
            {
              v9 = v107;
            }
LABEL_134:
            ;
          }
        }
      }
      ++v5;
      goto LABEL_115;
    case 1LL:
      goto LABEL_117;
  }
LABEL_112:
  v5 = v125;
LABEL_33:
  if ( v125 != v5 )
    v9 = 0;
  *a4 = v9;
  result = v123 + 72;
  v117 = v123 + 72;
  v121 = *(_QWORD *)(v123 + 80);
  if ( v121 != v123 + 72 )
  {
    v51 = a1;
    do
    {
      v56 = *(_DWORD *)(v51 + 64);
      v57 = v121 - 24;
      if ( !v121 )
        v57 = 0;
      if ( v56 )
      {
        v52 = v56 - 1;
        v53 = *(_QWORD *)(v51 + 48);
        v54 = (v56 - 1) & (((unsigned int)v57 >> 9) ^ ((unsigned int)v57 >> 4));
        v55 = *(_QWORD *)(v53 + 8LL * v54);
        if ( v57 == v55 )
          goto LABEL_38;
        v82 = 1;
        while ( v55 != -8 )
        {
          v54 = v52 & (v82 + v54);
          v55 = *(_QWORD *)(v53 + 8LL * v54);
          if ( v57 == v55 )
            goto LABEL_38;
          ++v82;
        }
      }
      v58 = *(_QWORD *)(v57 + 48);
      v131 = v57 + 40;
      if ( v57 + 40 == v58 )
        goto LABEL_38;
      while ( 2 )
      {
        if ( !v58 )
          BUG();
        if ( *(_BYTE *)(v58 - 8) != 53 )
          goto LABEL_46;
        v138[0] = v51;
        v135 = 0;
        v136 = 0;
        v138[1] = a4;
        v59 = (char *)sub_1ABB790((__int64)v138, v58 - 24, &v135, &v136);
        v122 = v59;
        v61 = v60;
        v124 = v60;
        if ( v59 )
        {
          if ( v135 )
          {
            v137 = v59;
            sub_1ABE500(a2, &v137);
          }
          v137 = (char *)(v58 - 24);
          sub_1ABE500(a2, &v137);
          if ( v136 )
          {
            v137 = v61;
            sub_1ABE500(a3, &v137);
          }
          goto LABEL_46;
        }
        v65 = *(_QWORD *)(v58 - 16);
        if ( !v65 )
          goto LABEL_46;
        v118 = 0;
        while ( 1 )
        {
          while ( 1 )
          {
            v73 = sub_1648700(v65);
            if ( v58 - 24 != sub_164A190((__int64)v73) )
              break;
            v135 = 0;
            v136 = 0;
            v122 = (char *)sub_1ABB790((__int64)v138, (__int64)v73, &v135, &v136);
            v124 = v74;
            if ( !v122 )
              break;
            v65 = *(_QWORD *)(v65 + 8);
            v118 = v73;
            if ( !v65 )
            {
LABEL_76:
              if ( !v118 )
                goto LABEL_46;
              if ( v135 )
              {
                v137 = v122;
                sub_1ABE500(a2, &v137);
              }
              if ( *((_BYTE *)v118 + 16) > 0x17u )
              {
                v75 = *(_DWORD *)(v51 + 64);
                if ( v75 )
                {
                  v76 = v118[5];
                  v77 = v75 - 1;
                  v78 = *(_QWORD *)(v51 + 48);
                  v79 = (v75 - 1) & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
                  v80 = *(_QWORD *)(v78 + 8LL * v79);
                  if ( v76 == v80 )
                    goto LABEL_82;
                  v81 = 1;
                  while ( v80 != -8 )
                  {
                    v79 = v77 & (v81 + v79);
                    v80 = *(_QWORD *)(v78 + 8LL * v79);
                    if ( v76 == v80 )
                      goto LABEL_82;
                    ++v81;
                  }
                }
              }
              v137 = (char *)v118;
              sub_1ABE500(a2, &v137);
LABEL_82:
              v137 = (char *)(v58 - 24);
              sub_1ABE500(a2, &v137);
              if ( v136 )
              {
                v137 = v124;
                sub_1ABE500(a3, &v137);
              }
              goto LABEL_46;
            }
          }
          if ( *((_BYTE *)v73 + 16) <= 0x17u )
            goto LABEL_46;
          v66 = *(_DWORD *)(v51 + 64);
          if ( !v66 )
            goto LABEL_46;
          v67 = v73[5];
          v68 = v66 - 1;
          v69 = *(_QWORD *)(v51 + 48);
          v70 = 1;
          v71 = (v66 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
          v72 = *(_QWORD *)(v69 + 8LL * (v68 & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4))));
          if ( v67 != v72 )
            break;
LABEL_72:
          v65 = *(_QWORD *)(v65 + 8);
          if ( !v65 )
            goto LABEL_76;
        }
        while ( v72 != -8 )
        {
          v71 = v68 & (v70 + v71);
          v72 = *(_QWORD *)(v69 + 8LL * v71);
          if ( v67 == v72 )
            goto LABEL_72;
          ++v70;
        }
LABEL_46:
        v58 = *(_QWORD *)(v58 + 8);
        if ( v131 != v58 )
          continue;
        break;
      }
LABEL_38:
      result = *(_QWORD *)(v121 + 8);
      v121 = result;
    }
    while ( v117 != result );
  }
  return result;
}
