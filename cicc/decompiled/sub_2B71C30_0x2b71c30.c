// Function: sub_2B71C30
// Address: 0x2b71c30
//
__int64 __fastcall sub_2B71C30(__int64 a1, const __m128i *a2, char a3)
{
  unsigned __int8 ***v4; // rax
  unsigned __int8 **v5; // rax
  unsigned int v6; // ebx
  __int64 v7; // rax
  bool v8; // zf
  __int64 *v9; // rax
  __int64 *v10; // rdx
  unsigned __int8 **v11; // r12
  unsigned __int8 **v12; // r15
  __int64 v13; // rbx
  __int64 v14; // r8
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rcx
  int v17; // eax
  _DWORD *v18; // rdx
  __int64 v19; // rbx
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rcx
  int v22; // eax
  unsigned __int8 **v23; // rdx
  unsigned __int8 *v24; // r13
  char v25; // al
  __int64 v26; // r9
  int v27; // ebx
  int v28; // esi
  __int64 v29; // r8
  unsigned int v30; // ecx
  __int64 v31; // rdx
  unsigned __int8 *v32; // r10
  __int64 v33; // rdi
  unsigned __int64 v34; // rcx
  unsigned __int64 v35; // rsi
  int v36; // edx
  _DWORD *v37; // rcx
  __int64 v38; // rdi
  __int64 v39; // rdx
  int v40; // eax
  unsigned __int8 **v41; // rdx
  __int64 v42; // rax
  unsigned __int64 v43; // r12
  __int64 v44; // rax
  unsigned int v45; // eax
  unsigned int v46; // ebx
  unsigned __int8 ***v47; // rax
  __int64 v48; // rdx
  char ***v49; // rcx
  __int64 v50; // rax
  __int64 v51; // rcx
  unsigned int v52; // eax
  unsigned int v53; // r13d
  unsigned __int8 ***v54; // rax
  unsigned __int8 **v55; // r8
  unsigned __int8 **v56; // r9
  unsigned __int8 ***v57; // rax
  unsigned int v58; // esi
  unsigned int v59; // eax
  __int64 v60; // rax
  unsigned __int8 **v61; // rcx
  __int64 v62; // r8
  __int64 v63; // rdi
  unsigned __int8 **v64; // rdx
  unsigned __int8 v65; // si
  unsigned int v67; // eax
  unsigned int v68; // edx
  int v69; // r11d
  int v70; // ecx
  __int64 *v71; // rsi
  unsigned int v72; // edx
  __int64 v73; // rdi
  __int64 v74; // rax
  unsigned __int8 v75; // si
  int v76; // ecx
  __int64 *v77; // rsi
  unsigned int v78; // edx
  __int64 v79; // rdi
  unsigned int v80; // r10d
  unsigned __int8 **v81; // r10
  __int64 v82; // rdx
  unsigned __int8 **v83; // rcx
  unsigned __int8 **v84; // rsi
  unsigned __int8 *v85; // rdi
  unsigned __int8 *v86; // rdi
  unsigned __int8 *v87; // rdi
  __int64 v88; // rax
  unsigned __int8 v89; // al
  __int64 v90; // rbx
  __int64 v91; // r15
  __int64 v92; // rdx
  unsigned __int64 v93; // r12
  unsigned __int64 v94; // rcx
  const void *v95; // r9
  __int64 v96; // rax
  __int64 v97; // r8
  __int64 v98; // r12
  __int64 v99; // rax
  __int64 v100; // rcx
  __int64 v101; // r8
  __int64 v102; // r9
  __int64 v103; // rdx
  __int64 v104; // rdx
  __int64 **v105; // rax
  __int64 *v106; // rcx
  __int64 *v107; // rdx
  unsigned __int8 v108; // si
  unsigned __int8 v109; // si
  unsigned int v110; // r10d
  unsigned __int8 v111; // al
  unsigned __int8 v112; // al
  char v113; // [rsp+7h] [rbp-169h]
  int v114; // [rsp+8h] [rbp-168h]
  const void *v116; // [rsp+10h] [rbp-160h]
  __int64 v118; // [rsp+20h] [rbp-150h]
  __int64 v119; // [rsp+30h] [rbp-140h] BYREF
  __int64 v120; // [rsp+38h] [rbp-138h]
  __int64 *v121; // [rsp+40h] [rbp-130h] BYREF
  unsigned int v122; // [rsp+48h] [rbp-128h]
  char v123; // [rsp+140h] [rbp-30h] BYREF

  v4 = *(unsigned __int8 ****)a1;
  v119 = 0;
  v5 = v4[1];
  if ( (unsigned int)v5 <= 0x10
    || (_BitScanReverse((unsigned int *)&v5, (_DWORD)v5 - 1), v6 = 1 << (32 - ((unsigned __int8)v5 ^ 0x1F)), v6 <= 0x10) )
  {
    v120 = 1;
  }
  else
  {
    LOBYTE(v120) = v120 & 0xFE;
    v7 = sub_C7D670(16LL * v6, 8);
    v8 = (v120 & 1) == 0;
    v120 &= 1u;
    v121 = (__int64 *)v7;
    v122 = v6;
    if ( v8 )
    {
      v9 = v121;
      v10 = &v121[2 * v122];
      if ( v121 == v10 )
        goto LABEL_5;
      goto LABEL_43;
    }
  }
  v10 = (__int64 *)&v123;
  v9 = (__int64 *)&v121;
  do
  {
LABEL_43:
    if ( v9 )
      *v9 = -4096;
    v9 += 2;
  }
  while ( v9 != v10 );
LABEL_5:
  v11 = **(unsigned __int8 ****)a1;
  v12 = &v11[*(_QWORD *)(*(_QWORD *)a1 + 8LL)];
  if ( v11 != v12 )
  {
    while ( 1 )
    {
      v24 = *v11;
      v25 = sub_2B0D8B0(*v11);
      if ( v25 )
      {
        v13 = *(_QWORD *)(a1 + 8);
        v14 = -1;
        if ( *v24 != 13 )
          v14 = *(unsigned int *)(*(_QWORD *)(a1 + 16) + 8LL);
        v15 = *(unsigned int *)(v13 + 8);
        v16 = *(unsigned int *)(v13 + 12);
        v17 = *(_DWORD *)(v13 + 8);
        if ( v15 >= v16 )
        {
          v26 = v15 + 1;
          if ( v16 < v15 + 1 )
          {
            v114 = v14;
            sub_C8D5F0(*(_QWORD *)(a1 + 8), (const void *)(v13 + 16), v15 + 1, 4u, v14, v26);
            v15 = *(unsigned int *)(v13 + 8);
            LODWORD(v14) = v114;
          }
          *(_DWORD *)(*(_QWORD *)v13 + 4 * v15) = v14;
          ++*(_DWORD *)(v13 + 8);
        }
        else
        {
          v18 = (_DWORD *)(*(_QWORD *)v13 + 4 * v15);
          if ( v18 )
          {
            *v18 = v14;
            v17 = *(_DWORD *)(v13 + 8);
          }
          *(_DWORD *)(v13 + 8) = v17 + 1;
        }
        v19 = *(_QWORD *)(a1 + 16);
        v20 = *(unsigned int *)(v19 + 8);
        v21 = *(unsigned int *)(v19 + 12);
        v22 = *(_DWORD *)(v19 + 8);
        if ( v20 >= v21 )
        {
          if ( v21 < v20 + 1 )
          {
            sub_C8D5F0(*(_QWORD *)(a1 + 16), (const void *)(v19 + 16), v20 + 1, 8u, v20 + 1, v26);
            v20 = *(unsigned int *)(v19 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v19 + 8 * v20) = v24;
          ++*(_DWORD *)(v19 + 8);
        }
        else
        {
          v23 = (unsigned __int8 **)(*(_QWORD *)v19 + 8 * v20);
          if ( v23 )
          {
            *v23 = v24;
            v22 = *(_DWORD *)(v19 + 8);
          }
          *(_DWORD *)(v19 + 8) = v22 + 1;
        }
        goto LABEL_17;
      }
      v27 = *(_DWORD *)(*(_QWORD *)(a1 + 16) + 8LL);
      if ( (v120 & 1) != 0 )
      {
        v28 = 15;
        v29 = (__int64)&v121;
      }
      else
      {
        v58 = v122;
        v29 = (__int64)v121;
        if ( !v122 )
        {
          v67 = v120;
          ++v119;
          v26 = 0;
          v68 = ((unsigned int)v120 >> 1) + 1;
          goto LABEL_63;
        }
        v28 = v122 - 1;
      }
      v30 = v28 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v31 = v29 + 16LL * v30;
      v32 = *(unsigned __int8 **)v31;
      if ( v24 == *(unsigned __int8 **)v31 )
      {
LABEL_22:
        v27 = *(_DWORD *)(v31 + 8);
        goto LABEL_23;
      }
      v69 = 1;
      v26 = 0;
      while ( v32 != (unsigned __int8 *)-4096LL )
      {
        if ( v32 == (unsigned __int8 *)-8192LL && !v26 )
          v26 = v31;
        v30 = v28 & (v69 + v30);
        v31 = v29 + 16LL * v30;
        v32 = *(unsigned __int8 **)v31;
        if ( v24 == *(unsigned __int8 **)v31 )
          goto LABEL_22;
        ++v69;
      }
      v67 = v120;
      if ( !v26 )
        v26 = v31;
      ++v119;
      v68 = ((unsigned int)v120 >> 1) + 1;
      if ( (v120 & 1) == 0 )
      {
        v58 = v122;
LABEL_63:
        if ( 4 * v68 < 3 * v58 )
          goto LABEL_64;
        goto LABEL_85;
      }
      v58 = 16;
      if ( 4 * v68 < 0x30 )
      {
LABEL_64:
        if ( v58 - HIDWORD(v120) - v68 > v58 >> 3 )
          goto LABEL_65;
        sub_2281F90((__int64)&v119, v58);
        if ( (v120 & 1) != 0 )
        {
          v76 = 15;
          v77 = (__int64 *)&v121;
        }
        else
        {
          v77 = v121;
          if ( !v122 )
          {
LABEL_182:
            LODWORD(v120) = (2 * ((unsigned int)v120 >> 1) + 2) | v120 & 1;
            BUG();
          }
          v76 = v122 - 1;
        }
        v67 = v120;
        v78 = v76 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v26 = (__int64)&v77[2 * v78];
        v79 = *(_QWORD *)v26;
        if ( v24 == *(unsigned __int8 **)v26 )
          goto LABEL_65;
        v29 = 1;
        v74 = 0;
        while ( v79 != -4096 )
        {
          if ( v79 == -8192 && !v74 )
            v74 = v26;
          v80 = v29 + 1;
          v29 = v78 + (unsigned int)v29;
          v78 = v76 & v29;
          v26 = (__int64)&v77[2 * (v76 & (unsigned int)v29)];
          v79 = *(_QWORD *)v26;
          if ( v24 == *(unsigned __int8 **)v26 )
            goto LABEL_92;
          v29 = v80;
        }
        goto LABEL_90;
      }
LABEL_85:
      sub_2281F90((__int64)&v119, 2 * v58);
      if ( (v120 & 1) != 0 )
      {
        v70 = 15;
        v71 = (__int64 *)&v121;
      }
      else
      {
        v71 = v121;
        if ( !v122 )
          goto LABEL_182;
        v70 = v122 - 1;
      }
      v67 = v120;
      v72 = v70 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v26 = (__int64)&v71[2 * v72];
      v73 = *(_QWORD *)v26;
      if ( v24 == *(unsigned __int8 **)v26 )
        goto LABEL_65;
      v29 = 1;
      v74 = 0;
      while ( v73 != -4096 )
      {
        if ( v73 == -8192 && !v74 )
          v74 = v26;
        v110 = v29 + 1;
        v29 = v72 + (unsigned int)v29;
        v72 = v70 & v29;
        v26 = (__int64)&v71[2 * (v70 & (unsigned int)v29)];
        v73 = *(_QWORD *)v26;
        if ( v24 == *(unsigned __int8 **)v26 )
          goto LABEL_92;
        v29 = v110;
      }
LABEL_90:
      if ( v74 )
        v26 = v74;
LABEL_92:
      v67 = v120;
LABEL_65:
      LODWORD(v120) = (2 * (v67 >> 1) + 2) | v67 & 1;
      if ( *(_QWORD *)v26 != -4096 )
        --HIDWORD(v120);
      *(_QWORD *)v26 = v24;
      v25 = 1;
      *(_DWORD *)(v26 + 8) = v27;
LABEL_23:
      v33 = *(_QWORD *)(a1 + 8);
      v34 = *(unsigned int *)(v33 + 8);
      v35 = *(unsigned int *)(v33 + 12);
      v36 = *(_DWORD *)(v33 + 8);
      if ( v34 >= v35 )
      {
        if ( v35 < v34 + 1 )
        {
          v113 = v25;
          sub_C8D5F0(v33, (const void *)(v33 + 16), v34 + 1, 4u, v29, v26);
          v25 = v113;
          v34 = *(unsigned int *)(v33 + 8);
        }
        *(_DWORD *)(*(_QWORD *)v33 + 4 * v34) = v27;
        ++*(_DWORD *)(v33 + 8);
      }
      else
      {
        v37 = (_DWORD *)(*(_QWORD *)v33 + 4 * v34);
        if ( v37 )
        {
          *v37 = v27;
          v36 = *(_DWORD *)(v33 + 8);
        }
        *(_DWORD *)(v33 + 8) = v36 + 1;
      }
      if ( !v25 )
        goto LABEL_17;
      v38 = *(_QWORD *)(a1 + 16);
      v39 = *(unsigned int *)(v38 + 8);
      v40 = v39;
      if ( *(_DWORD *)(v38 + 12) <= (unsigned int)v39 )
      {
        sub_94F890(v38, (__int64)v24);
LABEL_17:
        if ( v12 == ++v11 )
          break;
      }
      else
      {
        v41 = (unsigned __int8 **)(*(_QWORD *)v38 + 8 * v39);
        if ( v41 )
        {
          *v41 = v24;
          v40 = *(_DWORD *)(v38 + 8);
        }
        ++v11;
        *(_DWORD *)(v38 + 8) = v40 + 1;
        if ( v12 == v11 )
          break;
      }
    }
  }
  v42 = *(_QWORD *)(a1 + 16);
  v43 = *(unsigned int *)(v42 + 8);
  v44 = sub_2B08520(**(char ***)v42);
  LOBYTE(v45) = sub_2B1F720(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 3296LL), v44, v43);
  v46 = v45;
  v47 = *(unsigned __int8 ****)a1;
  v48 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
  if ( v43 == v48 )
  {
    v53 = v46;
    LOBYTE(v53) = qword_500F628 | v46;
    if ( (unsigned __int8)qword_500F628 | (unsigned __int8)v46 )
      goto LABEL_78;
  }
  v49 = **(char *****)(a1 + 32);
  if ( v49 )
  {
    v50 = sub_2B08520(**v49);
    if ( !sub_2B1F720(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 3296LL), v50, *(_DWORD *)(v51 + 8)) )
      goto LABEL_58;
    v47 = *(unsigned __int8 ****)a1;
    v48 = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
  }
  LOBYTE(v52) = sub_2B1F720(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 3296LL), *((_QWORD *)**v47 + 1), v48);
  v53 = v52;
  if ( !(_BYTE)v52 )
    goto LABEL_58;
  if ( v43 > 1 && (_BYTE)v46 == 1 )
  {
    v54 = *(unsigned __int8 ****)(a1 + 16);
    v55 = *v54;
    v56 = (unsigned __int8 **)*((unsigned int *)v54 + 2);
    if ( (unsigned int)v120 >> 1 != 1 )
    {
LABEL_40:
      v57 = *(unsigned __int8 ****)a1;
      *v57 = v55;
      v57[1] = v56;
      goto LABEL_59;
    }
    v81 = &v55[(_QWORD)v56];
    v82 = (8 * (__int64)v56) >> 3;
    if ( (8 * (__int64)v56) >> 5 )
    {
      v83 = *v54;
      v84 = &v55[4 * ((8 * (__int64)v56) >> 5)];
      while ( (unsigned int)**v83 - 12 <= 1 || !(unsigned __int8)sub_2B0D8B0(*v83) )
      {
        v85 = v83[1];
        if ( (unsigned int)*v85 - 12 > 1 && (unsigned __int8)sub_2B0D8B0(v85) )
        {
          ++v83;
          break;
        }
        v86 = v83[2];
        if ( (unsigned int)*v86 - 12 > 1 && (unsigned __int8)sub_2B0D8B0(v86) )
        {
          v83 += 2;
          break;
        }
        v87 = v83[3];
        if ( (unsigned int)*v87 - 12 > 1 && (unsigned __int8)sub_2B0D8B0(v87) )
        {
          v83 += 3;
          break;
        }
        v83 += 4;
        if ( v84 == v83 )
        {
          v82 = v81 - v83;
          goto LABEL_153;
        }
      }
LABEL_119:
      if ( v81 != v83 )
        goto LABEL_40;
LABEL_58:
      v53 = 0;
      sub_2B70420(
        *(_QWORD *)(a1 + 24),
        **(__int64 ***)a1,
        *(_QWORD *)(*(_QWORD *)a1 + 8LL),
        3,
        v118,
        a2,
        *(_DWORD **)(a1 + 32),
        0,
        0,
        0,
        0);
      goto LABEL_59;
    }
    v83 = *v54;
LABEL_153:
    if ( v82 != 2 )
    {
      if ( v82 != 3 )
      {
        if ( v82 != 1 )
          goto LABEL_58;
        goto LABEL_156;
      }
      if ( (unsigned int)**v83 - 12 > 1 && (unsigned __int8)sub_2B0D8B0(*v83) )
        goto LABEL_119;
      ++v83;
    }
    if ( (unsigned int)**v83 - 12 > 1 && (unsigned __int8)sub_2B0D8B0(*v83) )
      goto LABEL_119;
    ++v83;
LABEL_156:
    if ( (unsigned int)**v83 - 12 <= 1 || !(unsigned __int8)sub_2B0D8B0(*v83) )
      goto LABEL_58;
    goto LABEL_119;
  }
  if ( !a3 )
    goto LABEL_58;
  if ( (unsigned int)v120 <= 3 )
    goto LABEL_58;
  if ( v43 <= 1 )
    goto LABEL_58;
  LOBYTE(v59) = sub_B469C0((unsigned __int8 *)a2->m128i_i64[0]);
  v53 = v59;
  if ( !(_BYTE)v59 )
    goto LABEL_58;
  v60 = *(_QWORD *)(a1 + 16);
  v61 = *(unsigned __int8 ***)v60;
  v62 = *(unsigned int *)(v60 + 8);
  v63 = *(_QWORD *)v60 + 8 * v62;
  if ( !((8 * v62) >> 5) )
  {
    v64 = *(unsigned __int8 ***)v60;
LABEL_128:
    v88 = v63 - (_QWORD)v64;
    if ( v63 - (_QWORD)v64 != 16 )
    {
      if ( v88 != 24 )
      {
        if ( v88 != 8 )
          goto LABEL_133;
        goto LABEL_131;
      }
      v111 = **v64;
      if ( v111 != 13 && v111 <= 0x1Cu )
        goto LABEL_57;
      ++v64;
    }
    v112 = **v64;
    if ( v112 != 13 && v112 <= 0x1Cu )
      goto LABEL_57;
    ++v64;
LABEL_131:
    v89 = **v64;
    if ( v89 == 13 || v89 > 0x1Cu )
      goto LABEL_133;
    goto LABEL_57;
  }
  v64 = *(unsigned __int8 ***)v60;
  while ( 1 )
  {
    v65 = **v64;
    if ( v65 != 13 && v65 <= 0x1Cu )
      break;
    v75 = *v64[1];
    if ( v75 <= 0x1Cu && v75 != 13 )
    {
      ++v64;
      break;
    }
    v108 = *v64[2];
    if ( v108 != 13 && v108 <= 0x1Cu )
    {
      v64 += 2;
      break;
    }
    v109 = *v64[3];
    if ( v109 != 13 && v109 <= 0x1Cu )
    {
      v64 += 3;
      break;
    }
    v64 += 4;
    if ( &v61[4 * ((8 * v62) >> 5)] == v64 )
      goto LABEL_128;
  }
LABEL_57:
  if ( (unsigned __int8 **)v63 != v64 )
    goto LABEL_58;
LABEL_133:
  v90 = (unsigned int)sub_2B1E090(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 3296LL), *((_QWORD *)*v61 + 1), v62);
  if ( v90 == *(_QWORD *)(*(_QWORD *)a1 + 8LL) )
  {
LABEL_78:
    *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = 0;
    goto LABEL_59;
  }
  v96 = *(_QWORD *)(a1 + 16);
  v91 = *(_QWORD *)(a1 + 40);
  v92 = 0;
  v93 = *(unsigned int *)(v96 + 8);
  v94 = *(unsigned int *)(v91 + 12);
  v95 = *(const void **)v96;
  *(_DWORD *)(v91 + 8) = 0;
  LODWORD(v96) = 0;
  v97 = 8 * v93;
  if ( v93 > v94 )
  {
    v116 = v95;
    sub_C8D5F0(v91, (const void *)(v91 + 16), v93, 8u, v97, (__int64)v95);
    v96 = *(unsigned int *)(v91 + 8);
    v97 = 8 * v93;
    v95 = v116;
    v92 = 8 * v96;
  }
  if ( v97 )
  {
    memcpy((void *)(v92 + *(_QWORD *)v91), v95, v97);
    LODWORD(v96) = *(_DWORD *)(v91 + 8);
  }
  *(_DWORD *)(v91 + 8) = v96 + v93;
  v98 = *(_QWORD *)(a1 + 40);
  v99 = sub_ACADE0(*(__int64 ***)(***(_QWORD ***)(a1 + 16) + 8LL));
  sub_2B3B580(v98, v90 - *(unsigned int *)(*(_QWORD *)(a1 + 16) + 8LL), v99, v100, v101, v102);
  if ( sub_2B5F980(
         **(__int64 ***)(a1 + 40),
         *(unsigned int *)(*(_QWORD *)(a1 + 40) + 8LL),
         *(__int64 **)(*(_QWORD *)(a1 + 24) + 3304LL))
    && v103 )
  {
    v104 = *(_QWORD *)(a1 + 40);
    v105 = *(__int64 ***)a1;
    v106 = *(__int64 **)v104;
    v107 = (__int64 *)*(unsigned int *)(v104 + 8);
    *v105 = v106;
    v105[1] = v107;
  }
  else
  {
    v53 = 0;
    sub_2B71BE0(
      *(_QWORD *)(a1 + 24),
      **(__int64 ***)a1,
      *(_QWORD *)(*(_QWORD *)a1 + 8LL),
      0,
      v118,
      a2,
      *(_DWORD **)(a1 + 32),
      0,
      0,
      0,
      0,
      0);
  }
LABEL_59:
  if ( (v120 & 1) == 0 )
    sub_C7D6A0((__int64)v121, 16LL * v122, 8);
  return v53;
}
