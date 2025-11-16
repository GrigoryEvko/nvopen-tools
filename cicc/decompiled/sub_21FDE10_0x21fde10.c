// Function: sub_21FDE10
// Address: 0x21fde10
//
__int64 *__fastcall sub_21FDE10(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5)
{
  int v5; // r10d
  __int64 v6; // r9
  __int64 v10; // rax
  __int16 v11; // dx
  bool v12; // al
  __int64 v13; // rax
  __int16 v14; // dx
  int v15; // edx
  int v16; // r10d
  __int64 *result; // rax
  int v18; // eax
  __int64 v19; // rdx
  int v20; // r14d
  unsigned int v21; // ecx
  __int64 v22; // r15
  __int64 v23; // rbx
  __int64 v24; // rax
  int v25; // esi
  __int64 v26; // rdx
  int v27; // eax
  __int64 v28; // r8
  unsigned int v29; // esi
  _DWORD *v30; // rcx
  int v31; // edi
  __int64 v32; // rax
  __int16 v33; // dx
  bool v34; // al
  bool v35; // al
  __int64 v36; // r15
  __int64 v37; // rbx
  int v38; // r14d
  __int64 v39; // rcx
  __int64 v40; // rax
  __int64 v41; // rdx
  int v42; // eax
  __int64 v43; // rdi
  unsigned int v44; // esi
  int *v45; // rcx
  int v46; // r8d
  unsigned int v47; // esi
  int v48; // eax
  __int64 v49; // r8
  unsigned int v50; // ecx
  int *v51; // rdx
  int v52; // edi
  unsigned int v53; // esi
  __int64 v54; // rdi
  __int64 v55; // rcx
  int v56; // r13d
  __int64 v57; // rdx
  int *v58; // rax
  int v59; // r9d
  __int64 v60; // rdi
  int v61; // eax
  int v62; // edx
  __int64 v63; // rsi
  unsigned int v64; // eax
  int v65; // ecx
  int v66; // edi
  int v67; // ecx
  int v68; // r10d
  int v69; // ecx
  int v70; // r11d
  int v71; // ebx
  int *v72; // r11
  int v73; // edi
  int v74; // edi
  int v75; // r14d
  int *v76; // rbx
  int v77; // eax
  int v78; // edx
  __int64 v79; // rax
  int v80; // r8d
  int v81; // r8d
  __int64 v82; // r9
  unsigned int v83; // eax
  int v84; // edi
  int v85; // esi
  int *v86; // rcx
  int v87; // edx
  int v88; // r9d
  __int64 v89; // r11
  __int64 v90; // r15
  int v91; // esi
  int v92; // ecx
  int *v93; // r8
  int v94; // edx
  int v95; // r9d
  __int64 v96; // r11
  int v97; // ecx
  __int64 v98; // r15
  int v99; // esi
  int v100; // esi
  int v101; // esi
  __int64 v102; // r8
  int v103; // ecx
  unsigned int v104; // r13d
  int *v105; // rax
  int v106; // edi
  int v107; // [rsp+0h] [rbp-90h]
  unsigned __int8 v108; // [rsp+7h] [rbp-89h]
  char v109; // [rsp+8h] [rbp-88h]
  _DWORD *v110; // [rsp+8h] [rbp-88h]
  int v111; // [rsp+8h] [rbp-88h]
  int v112; // [rsp+8h] [rbp-88h]
  _DWORD *v113; // [rsp+8h] [rbp-88h]
  int v114; // [rsp+8h] [rbp-88h]
  int v115; // [rsp+8h] [rbp-88h]
  int v117; // [rsp+10h] [rbp-80h]
  int v118; // [rsp+10h] [rbp-80h]
  __int64 v119; // [rsp+18h] [rbp-78h] BYREF
  int v120; // [rsp+2Ch] [rbp-64h] BYREF
  char v121[96]; // [rsp+30h] [rbp-60h] BYREF

  v6 = a2;
  v10 = *(_QWORD *)(a2 + 16);
  v119 = a2;
  v108 = *(_BYTE *)(v10 + 4);
  if ( *(_WORD *)v10 == 1 && (*(_BYTE *)(*(_QWORD *)(a2 + 32) + 64LL) & 0x10) != 0 )
    goto LABEL_21;
  v11 = *(_WORD *)(a2 + 46);
  if ( (v11 & 4) != 0 || (v11 & 8) == 0 )
  {
    v12 = (*(_QWORD *)(v10 + 8) & 0x20000LL) != 0;
  }
  else
  {
    v111 = v5;
    v12 = sub_1E15D00(a2, 0x20000u, 1);
    v6 = v119;
    v5 = v111;
  }
  if ( v12
    || (v13 = *(_QWORD *)(v6 + 16), *(_WORD *)v13 == 1) && (*(_BYTE *)(*(_QWORD *)(v6 + 32) + 64LL) & 8) != 0
    || ((v14 = *(_WORD *)(v6 + 46), (v14 & 4) != 0) || (v14 & 8) == 0
      ? (v109 = WORD1(*(_QWORD *)(v13 + 8)) & 1)
      : (v107 = v5, v35 = sub_1E15D00(v6, 0x10000u, 1), v6 = v119, v5 = v107, v109 = v35),
        v109) )
  {
LABEL_21:
    v18 = sub_21F8260(a1, v6);
    v19 = v119;
    v20 = v18;
    v21 = *(_DWORD *)(v119 + 40);
    if ( !v21 )
      goto LABEL_18;
    v22 = v21;
    v23 = 0;
    while ( 1 )
    {
      v24 = *(_QWORD *)(v19 + 32) + 40 * v23;
      if ( *(_BYTE *)v24 )
      {
        if ( *(_BYTE *)v24 == 5 && (!v20 || v20 + 5 != (_DWORD)v23) )
        {
          v25 = *(_DWORD *)(v24 + 24);
LABEL_31:
          sub_21FB820(a1, v25);
        }
      }
      else if ( (*(_BYTE *)(v24 + 3) & 0x10) == 0 )
      {
        v26 = *(unsigned int *)(a3 + 24);
        if ( (_DWORD)v26 )
        {
          v27 = *(_DWORD *)(v24 + 8);
          v28 = *(_QWORD *)(a3 + 8);
          v29 = (v26 - 1) & (37 * v27);
          v30 = (_DWORD *)(v28 + 8LL * v29);
          v31 = *v30;
          if ( v27 == *v30 )
          {
LABEL_34:
            if ( v30 != (_DWORD *)(v28 + 8 * v26) )
            {
              if ( v20 && v20 + 5 == (_DWORD)v23
                || (v110 = v30,
                    sub_21FB820(a1, v30[1]),
                    v30 = v110,
                    v110 != (_DWORD *)(*(_QWORD *)(a3 + 8) + 8LL * *(unsigned int *)(a3 + 24))) )
              {
                v25 = v30[1];
                if ( v25 >= (int)(-858993459
                                * ((__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 472) + 16LL)
                                           - *(_QWORD *)(*(_QWORD *)(a1 + 472) + 8LL)) >> 3)) )
                {
                  v32 = *(_QWORD *)(v119 + 16);
                  if ( *(_WORD *)v32 == 1 && (*(_BYTE *)(*(_QWORD *)(v119 + 32) + 64LL) & 0x10) != 0 )
                    goto LABEL_31;
                  v33 = *(_WORD *)(v119 + 46);
                  if ( (v33 & 4) == 0 && (v33 & 8) != 0 )
                  {
                    v113 = v30;
                    v34 = sub_1E15D00(v119, 0x20000u, 1);
                    v30 = v113;
                  }
                  else
                  {
                    v34 = (*(_QWORD *)(v32 + 8) & 0x20000LL) != 0;
                  }
                  if ( v34 || *(_BYTE *)(v119 + 49) != 1 )
                  {
                    v25 = v30[1];
                    goto LABEL_31;
                  }
                }
              }
            }
          }
          else
          {
            v67 = 1;
            while ( v31 != -1 )
            {
              v68 = v67 + 1;
              v29 = (v26 - 1) & (v67 + v29);
              v30 = (_DWORD *)(v28 + 8LL * v29);
              v31 = *v30;
              if ( v27 == *v30 )
                goto LABEL_34;
              v67 = v68;
            }
          }
        }
      }
      if ( v22 == ++v23 )
        goto LABEL_18;
      v19 = v119;
    }
  }
  if ( !a4 )
  {
    v15 = **(unsigned __int16 **)(v6 + 16);
    if ( (unsigned int)(v15 - 3224) <= 1 || (unsigned __int16)(v15 - 3221) <= 1u )
    {
      v16 = -858993459
          * ((__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 472) + 16LL) - *(_QWORD *)(*(_QWORD *)(a1 + 472) + 8LL)) >> 3)
          + sub_21D79A0(*(_QWORD *)(a1 + 504), *(char **)(*(_QWORD *)(v6 + 32) + 64LL));
      goto LABEL_15;
    }
  }
  v36 = *(unsigned int *)(v6 + 40);
  if ( !(_DWORD)v36 )
    goto LABEL_18;
  v37 = 0;
  v38 = v5;
  while ( 1 )
  {
    v39 = *(_QWORD *)(v6 + 32);
    v40 = v39 + 40 * v37;
    if ( *(_BYTE *)v40 )
    {
      if ( *(_BYTE *)v40 != 5 )
        goto LABEL_54;
      v38 = *(_DWORD *)(v40 + 24);
      if ( (unsigned int)**(unsigned __int16 **)(v6 + 16) - 3073 > 1 )
        goto LABEL_61;
      if ( v37 == 1 )
      {
        if ( *(_BYTE *)(v39 + 80) != 1 || *(_QWORD *)(v39 + 104) )
          goto LABEL_61;
      }
      else if ( (_DWORD)v37 != 2 || *(_BYTE *)(v39 + 40) != 1 || *(_QWORD *)(v39 + 64) )
      {
        goto LABEL_61;
      }
LABEL_69:
      v109 = 1;
      goto LABEL_54;
    }
    if ( (*(_BYTE *)(v40 + 3) & 0x10) == 0 )
    {
      v41 = *(unsigned int *)(a3 + 24);
      if ( (_DWORD)v41 )
        break;
    }
LABEL_54:
    if ( v36 == ++v37 )
      goto LABEL_62;
LABEL_55:
    v6 = v119;
  }
  v42 = *(_DWORD *)(v40 + 8);
  v43 = *(_QWORD *)(a3 + 8);
  v44 = (v41 - 1) & (37 * v42);
  v45 = (int *)(v43 + 8LL * v44);
  v46 = *v45;
  if ( v42 != *v45 )
  {
    v69 = 1;
    while ( v46 != -1 )
    {
      v70 = v69 + 1;
      v44 = (v41 - 1) & (v44 + v69);
      v45 = (int *)(v43 + 8LL * v44);
      v46 = *v45;
      if ( v42 == *v45 )
        goto LABEL_66;
      v69 = v70;
    }
    goto LABEL_54;
  }
LABEL_66:
  if ( v45 == (int *)(v43 + 8 * v41) )
    goto LABEL_54;
  v38 = v45[1];
  if ( (unsigned __int16)(**(_WORD **)(v6 + 16) - 4852) <= 0xEu && ((1LL << (**(_BYTE **)(v6 + 16) + 12)) & 0x5003) != 0 )
    goto LABEL_69;
LABEL_61:
  ++v37;
  sub_21FB820(a1, v38);
  if ( v36 != v37 )
    goto LABEL_55;
LABEL_62:
  v16 = v38;
  if ( !v109 )
    goto LABEL_18;
LABEL_15:
  if ( v16 != -1 )
  {
    if ( !*(_DWORD *)(a1 + 344) )
      goto LABEL_17;
    v61 = *(_DWORD *)(a1 + 352);
    if ( !v61 )
      goto LABEL_17;
    v62 = v61 - 1;
    v63 = *(_QWORD *)(a1 + 336);
    v64 = (v61 - 1) & (37 * v16);
    v65 = *(_DWORD *)(v63 + 4LL * v64);
    if ( v16 != v65 )
    {
      v66 = 1;
      while ( v65 != -1 )
      {
        v64 = v62 & (v66 + v64);
        v65 = *(_DWORD *)(v63 + 4LL * v64);
        if ( v16 == v65 )
          goto LABEL_18;
        ++v66;
      }
LABEL_17:
      if ( v108 <= 1u )
      {
        v47 = *(_DWORD *)(a3 + 24);
        v48 = *(_DWORD *)(*(_QWORD *)(v119 + 32) + 8LL);
        v120 = v48;
        if ( v47 )
        {
          v49 = *(_QWORD *)(a3 + 8);
          v50 = (v47 - 1) & (37 * v48);
          v51 = (int *)(v49 + 8LL * v50);
          v52 = *v51;
          if ( v48 == *v51 )
            goto LABEL_72;
          v71 = 1;
          v72 = 0;
          while ( v52 != -1 )
          {
            if ( v52 == -2 && !v72 )
              v72 = v51;
            v50 = (v47 - 1) & (v71 + v50);
            v51 = (int *)(v49 + 8LL * v50);
            v52 = *v51;
            if ( v48 == *v51 )
              goto LABEL_72;
            ++v71;
          }
          v73 = *(_DWORD *)(a3 + 16);
          if ( v72 )
            v51 = v72;
          ++*(_QWORD *)a3;
          v74 = v73 + 1;
          if ( 4 * v74 < 3 * v47 )
          {
            if ( v47 - *(_DWORD *)(a3 + 20) - v74 > v47 >> 3 )
            {
LABEL_105:
              *(_DWORD *)(a3 + 16) = v74;
              if ( *v51 != -1 )
                --*(_DWORD *)(a3 + 20);
              *v51 = v48;
              v51[1] = 0;
LABEL_72:
              v51[1] = v16;
              if ( a5 )
              {
                v112 = v16;
                sub_217F7B0((__int64)v121, a5, &v120);
                v16 = v112;
              }
              v53 = *(_DWORD *)(a1 + 384);
              v54 = a1 + 360;
              if ( v53 )
              {
                v55 = *(_QWORD *)(a1 + 368);
                v56 = 37 * v16;
                LODWORD(v57) = (v53 - 1) & (37 * v16);
                v58 = (int *)(v55 + 16LL * (unsigned int)v57);
                v59 = *v58;
                if ( *v58 == v16 )
                {
LABEL_76:
                  v60 = *((_QWORD *)v58 + 1);
                  if ( v60 )
                  {
LABEL_77:
                    result = sub_21FDBD0(v60, &v119);
                    *((_DWORD *)result + 2) = 1;
                    return result;
                  }
                  v76 = v58;
LABEL_117:
                  v79 = sub_22077B0(32);
                  v60 = v79;
                  if ( v79 )
                  {
                    *(_QWORD *)v79 = 0;
                    *(_QWORD *)(v79 + 8) = 0;
                    *(_QWORD *)(v79 + 16) = 0;
                    *(_DWORD *)(v79 + 24) = 0;
                  }
                  *((_QWORD *)v76 + 1) = v79;
                  goto LABEL_77;
                }
                v75 = 1;
                v76 = 0;
                while ( v59 != 0x7FFFFFFF )
                {
                  if ( v59 == 0x80000000 && !v76 )
                    v76 = v58;
                  v57 = (v53 - 1) & ((_DWORD)v57 + v75);
                  v58 = (int *)(v55 + 16 * v57);
                  v59 = *v58;
                  if ( *v58 == v16 )
                    goto LABEL_76;
                  ++v75;
                }
                if ( !v76 )
                  v76 = v58;
                v77 = *(_DWORD *)(a1 + 376);
                ++*(_QWORD *)(a1 + 360);
                v78 = v77 + 1;
                if ( 4 * (v77 + 1) < 3 * v53 )
                {
                  if ( v53 - *(_DWORD *)(a1 + 380) - v78 > v53 >> 3 )
                  {
LABEL_114:
                    *(_DWORD *)(a1 + 376) = v78;
                    if ( *v76 != 0x7FFFFFFF )
                      --*(_DWORD *)(a1 + 380);
                    *v76 = v16;
                    *((_QWORD *)v76 + 1) = 0;
                    goto LABEL_117;
                  }
                  v118 = v16;
                  sub_21FDA00(v54, v53);
                  v100 = *(_DWORD *)(a1 + 384);
                  if ( v100 )
                  {
                    v101 = v100 - 1;
                    v102 = *(_QWORD *)(a1 + 368);
                    v16 = v118;
                    v103 = 1;
                    v104 = v101 & v56;
                    v78 = *(_DWORD *)(a1 + 376) + 1;
                    v105 = 0;
                    v76 = (int *)(v102 + 16LL * v104);
                    v106 = *v76;
                    if ( v118 != *v76 )
                    {
                      while ( v106 != 0x7FFFFFFF )
                      {
                        if ( !v105 && v106 == 0x80000000 )
                          v105 = v76;
                        v104 = v101 & (v103 + v104);
                        v76 = (int *)(v102 + 16LL * v104);
                        v106 = *v76;
                        if ( *v76 == v118 )
                          goto LABEL_114;
                        ++v103;
                      }
                      if ( v105 )
                        v76 = v105;
                    }
                    goto LABEL_114;
                  }
LABEL_176:
                  ++*(_DWORD *)(a1 + 376);
                  BUG();
                }
              }
              else
              {
                ++*(_QWORD *)(a1 + 360);
              }
              v117 = v16;
              sub_21FDA00(v54, 2 * v53);
              v80 = *(_DWORD *)(a1 + 384);
              if ( v80 )
              {
                v16 = v117;
                v81 = v80 - 1;
                v82 = *(_QWORD *)(a1 + 368);
                v78 = *(_DWORD *)(a1 + 376) + 1;
                v83 = v81 & (37 * v117);
                v76 = (int *)(v82 + 16LL * v83);
                v84 = *v76;
                if ( *v76 != v117 )
                {
                  v85 = 1;
                  v86 = 0;
                  while ( v84 != 0x7FFFFFFF )
                  {
                    if ( v84 == 0x80000000 && !v86 )
                      v86 = v76;
                    v83 = v81 & (v85 + v83);
                    v76 = (int *)(v82 + 16LL * v83);
                    v84 = *v76;
                    if ( *v76 == v117 )
                      goto LABEL_114;
                    ++v85;
                  }
                  if ( v86 )
                    v76 = v86;
                }
                goto LABEL_114;
              }
              goto LABEL_176;
            }
            v115 = v16;
            sub_1BFDD60(a3, v47);
            v94 = *(_DWORD *)(a3 + 24);
            if ( v94 )
            {
              v48 = v120;
              v95 = v94 - 1;
              v96 = *(_QWORD *)(a3 + 8);
              v93 = 0;
              v16 = v115;
              v74 = *(_DWORD *)(a3 + 16) + 1;
              v97 = 1;
              v98 = (v94 - 1) & (unsigned int)(37 * v120);
              v51 = (int *)(v96 + 8 * v98);
              v99 = *v51;
              if ( *v51 == v120 )
                goto LABEL_105;
              while ( v99 != -1 )
              {
                if ( !v93 && v99 == -2 )
                  v93 = v51;
                LODWORD(v98) = v95 & (v97 + v98);
                v51 = (int *)(v96 + 8LL * (unsigned int)v98);
                v99 = *v51;
                if ( v120 == *v51 )
                  goto LABEL_105;
                ++v97;
              }
              goto LABEL_133;
            }
            goto LABEL_177;
          }
        }
        else
        {
          ++*(_QWORD *)a3;
        }
        v114 = v16;
        sub_1BFDD60(a3, 2 * v47);
        v87 = *(_DWORD *)(a3 + 24);
        if ( v87 )
        {
          v48 = v120;
          v88 = v87 - 1;
          v89 = *(_QWORD *)(a3 + 8);
          v16 = v114;
          v74 = *(_DWORD *)(a3 + 16) + 1;
          v90 = (v87 - 1) & (unsigned int)(37 * v120);
          v51 = (int *)(v89 + 8 * v90);
          v91 = *v51;
          if ( *v51 == v120 )
            goto LABEL_105;
          v92 = 1;
          v93 = 0;
          while ( v91 != -1 )
          {
            if ( v91 == -2 && !v93 )
              v93 = v51;
            LODWORD(v90) = v88 & (v92 + v90);
            v51 = (int *)(v89 + 8LL * (unsigned int)v90);
            v91 = *v51;
            if ( v120 == *v51 )
              goto LABEL_105;
            ++v92;
          }
LABEL_133:
          if ( v93 )
            v51 = v93;
          goto LABEL_105;
        }
LABEL_177:
        ++*(_DWORD *)(a3 + 16);
        BUG();
      }
    }
  }
LABEL_18:
  result = (__int64 *)a5;
  if ( a5 )
    return (__int64 *)sub_21FC450(a1, v119, a3, a5);
  return result;
}
