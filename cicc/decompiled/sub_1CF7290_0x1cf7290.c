// Function: sub_1CF7290
// Address: 0x1cf7290
//
int *__fastcall sub_1CF7290(int *a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  __int64 *v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rsi
  __int64 *v16; // rdi
  unsigned __int64 v17; // rbx
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rdx
  __int64 *v21; // rax
  char v22; // r8
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // r8
  unsigned __int64 v25; // rbx
  __int64 v26; // rax
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rax
  char v30; // si
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rbx
  unsigned __int64 v33; // rdx
  _QWORD *v34; // r13
  __int64 v35; // r12
  __int64 v36; // r11
  unsigned int v37; // ecx
  _QWORD *v38; // rax
  __int64 v39; // rdi
  int v40; // r14d
  __int64 v41; // rbx
  unsigned int v42; // esi
  int v43; // r8d
  int v44; // r8d
  __int64 v45; // r9
  unsigned int v46; // eax
  int v47; // ecx
  _QWORD *v48; // rdx
  __int64 v49; // r13
  __int64 *v50; // rax
  char v51; // dl
  __int64 v52; // r12
  __int64 *v53; // rax
  __int64 *v54; // rcx
  __int64 *v55; // rsi
  unsigned __int64 v56; // rcx
  char v57; // al
  char v58; // si
  bool v59; // al
  int v61; // r10d
  int v62; // eax
  int v63; // r8d
  int v64; // r8d
  __int64 v65; // r9
  int v66; // esi
  unsigned int v67; // r13d
  _QWORD *v68; // rax
  __int64 v69; // rdi
  int v70; // edi
  _QWORD *v71; // rsi
  __int64 v72; // [rsp+0h] [rbp-350h]
  __int64 v73; // [rsp+0h] [rbp-350h]
  __int64 v74; // [rsp+0h] [rbp-350h]
  __int64 v75; // [rsp+8h] [rbp-348h]
  _QWORD v76[16]; // [rsp+20h] [rbp-330h] BYREF
  __int64 v77; // [rsp+A0h] [rbp-2B0h] BYREF
  _QWORD *v78; // [rsp+A8h] [rbp-2A8h]
  _QWORD *v79; // [rsp+B0h] [rbp-2A0h]
  __int64 v80; // [rsp+B8h] [rbp-298h]
  int v81; // [rsp+C0h] [rbp-290h]
  _QWORD v82[8]; // [rsp+C8h] [rbp-288h] BYREF
  unsigned __int64 v83; // [rsp+108h] [rbp-248h] BYREF
  unsigned __int64 v84; // [rsp+110h] [rbp-240h]
  unsigned __int64 v85; // [rsp+118h] [rbp-238h]
  __int64 v86; // [rsp+120h] [rbp-230h] BYREF
  __int64 *v87; // [rsp+128h] [rbp-228h]
  __int64 *v88; // [rsp+130h] [rbp-220h]
  unsigned int v89; // [rsp+138h] [rbp-218h]
  unsigned int v90; // [rsp+13Ch] [rbp-214h]
  int v91; // [rsp+140h] [rbp-210h]
  _BYTE v92[64]; // [rsp+148h] [rbp-208h] BYREF
  unsigned __int64 v93; // [rsp+188h] [rbp-1C8h] BYREF
  unsigned __int64 v94; // [rsp+190h] [rbp-1C0h]
  unsigned __int64 v95; // [rsp+198h] [rbp-1B8h]
  __int64 v96; // [rsp+1A0h] [rbp-1B0h] BYREF
  __int64 v97; // [rsp+1A8h] [rbp-1A8h]
  unsigned __int64 v98; // [rsp+1B0h] [rbp-1A0h]
  _BYTE v99[64]; // [rsp+1C8h] [rbp-188h] BYREF
  unsigned __int64 v100; // [rsp+208h] [rbp-148h]
  unsigned __int64 v101; // [rsp+210h] [rbp-140h]
  unsigned __int64 v102; // [rsp+218h] [rbp-138h]
  _QWORD v103[2]; // [rsp+220h] [rbp-130h] BYREF
  unsigned __int64 v104; // [rsp+230h] [rbp-120h]
  char v105[64]; // [rsp+248h] [rbp-108h] BYREF
  __int64 *v106; // [rsp+288h] [rbp-C8h]
  __int64 *v107; // [rsp+290h] [rbp-C0h]
  unsigned __int64 v108; // [rsp+298h] [rbp-B8h]
  _QWORD v109[2]; // [rsp+2A0h] [rbp-B0h] BYREF
  unsigned __int64 v110; // [rsp+2B0h] [rbp-A0h]
  char v111[64]; // [rsp+2C8h] [rbp-88h] BYREF
  unsigned __int64 v112; // [rsp+308h] [rbp-48h]
  unsigned __int64 v113; // [rsp+310h] [rbp-40h]
  unsigned __int64 v114; // [rsp+318h] [rbp-38h]

  *a1 = 1;
  *((_QWORD *)a1 + 1) = 0;
  *((_QWORD *)a1 + 2) = 0;
  *((_QWORD *)a1 + 3) = 0;
  a1[8] = 0;
  v75 = (__int64)(a1 + 2);
  memset(v76, 0, sizeof(v76));
  v78 = v82;
  v76[1] = &v76[5];
  v76[2] = &v76[5];
  v3 = *(_QWORD *)(a2 + 56);
  v80 = 0x100000008LL;
  v82[0] = v3;
  v103[0] = v3;
  v79 = v82;
  LODWORD(v76[3]) = 8;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v81 = 0;
  v77 = 1;
  LOBYTE(v104) = 0;
  sub_13B8390(&v83, (__int64)v103);
  sub_16CCEE0(&v96, (__int64)v99, 8, (__int64)v76);
  v4 = v76[13];
  memset(&v76[13], 0, 24);
  v100 = v4;
  v101 = v76[14];
  v102 = v76[15];
  sub_16CCEE0(&v86, (__int64)v92, 8, (__int64)&v77);
  v5 = v83;
  v83 = 0;
  v93 = v5;
  v6 = v84;
  v84 = 0;
  v94 = v6;
  v7 = v85;
  v85 = 0;
  v95 = v7;
  sub_16CCEE0(v103, (__int64)v105, 8, (__int64)&v86);
  v8 = v93;
  v93 = 0;
  v106 = (__int64 *)v8;
  v9 = (__int64 *)v94;
  v94 = 0;
  v107 = v9;
  v10 = v95;
  v95 = 0;
  v108 = v10;
  sub_16CCEE0(v109, (__int64)v111, 8, (__int64)&v96);
  v11 = v100;
  v100 = 0;
  v112 = v11;
  v12 = v101;
  v101 = 0;
  v113 = v12;
  v13 = v102;
  v102 = 0;
  v114 = v13;
  if ( v93 )
    j_j___libc_free_0(v93, v95 - v93);
  if ( v88 != v87 )
    _libc_free((unsigned __int64)v88);
  if ( v100 )
    j_j___libc_free_0(v100, v102 - v100);
  if ( v98 != v97 )
    _libc_free(v98);
  if ( v83 )
    j_j___libc_free_0(v83, v85 - v83);
  if ( v79 != v78 )
    _libc_free((unsigned __int64)v79);
  if ( v76[13] )
    j_j___libc_free_0(v76[13], v76[15] - v76[13]);
  if ( v76[2] != v76[1] )
    _libc_free(v76[2]);
  sub_16CCCB0(&v86, (__int64)v92, (__int64)v103);
  v15 = v107;
  v16 = v106;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v17 = (char *)v107 - (char *)v106;
  if ( v107 == v106 )
  {
    v17 = 0;
    v19 = 0;
  }
  else
  {
    if ( v17 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_129;
    v18 = sub_22077B0((char *)v107 - (char *)v106);
    v15 = v107;
    v16 = v106;
    v19 = v18;
  }
  v93 = v19;
  v94 = v19;
  v95 = v19 + v17;
  if ( v16 != v15 )
  {
    v20 = v19;
    v21 = v16;
    do
    {
      if ( v20 )
      {
        *(_QWORD *)v20 = *v21;
        v22 = *((_BYTE *)v21 + 16);
        *(_BYTE *)(v20 + 16) = v22;
        if ( v22 )
          *(_QWORD *)(v20 + 8) = v21[1];
      }
      v21 += 3;
      v20 += 24LL;
    }
    while ( v21 != v15 );
    v19 += 8 * ((unsigned __int64)((char *)(v21 - 3) - (char *)v16) >> 3) + 24;
  }
  v15 = (__int64 *)v99;
  v16 = &v96;
  v94 = v19;
  sub_16CCCB0(&v96, (__int64)v99, (__int64)v109);
  v23 = v113;
  v24 = v112;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v25 = v113 - v112;
  if ( v113 != v112 )
  {
    if ( v25 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v26 = sub_22077B0(v113 - v112);
      v23 = v113;
      v24 = v112;
      v27 = v26;
      goto LABEL_30;
    }
LABEL_129:
    sub_4261EA(v16, v15, v14);
  }
  v27 = 0;
LABEL_30:
  v100 = v27;
  v101 = v27;
  v102 = v27 + v25;
  if ( v23 == v24 )
  {
    v31 = v27;
  }
  else
  {
    v28 = v27;
    v29 = v24;
    do
    {
      if ( v28 )
      {
        *(_QWORD *)v28 = *(_QWORD *)v29;
        v30 = *(_BYTE *)(v29 + 16);
        *(_BYTE *)(v28 + 16) = v30;
        if ( v30 )
          *(_QWORD *)(v28 + 8) = *(_QWORD *)(v29 + 8);
      }
      v29 += 24LL;
      v28 += 24LL;
    }
    while ( v23 != v29 );
    v31 = v27 + 8 * ((v23 - 24 - v24) >> 3) + 24;
  }
  v32 = v94;
  v33 = v93;
  v101 = v31;
  if ( v94 - v93 == v31 - v27 )
    goto LABEL_67;
  do
  {
LABEL_38:
    v34 = *(_QWORD **)(v32 - 24);
    v35 = *(_QWORD *)(*v34 + 48LL);
    v36 = *v34 + 40LL;
    if ( v36 == v35 )
      goto LABEL_52;
    do
    {
      while ( 1 )
      {
        v40 = *a1;
        v41 = v35 - 24;
        v42 = a1[8];
        if ( !v35 )
          v41 = 0;
        *a1 = v40 + 1;
        if ( !v42 )
        {
          ++*((_QWORD *)a1 + 1);
LABEL_46:
          v72 = v36;
          sub_1541C50(v75, 2 * v42);
          v43 = a1[8];
          if ( !v43 )
            goto LABEL_135;
          v44 = v43 - 1;
          v45 = *((_QWORD *)a1 + 2);
          v36 = v72;
          v46 = v44 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
          v47 = a1[6] + 1;
          v48 = (_QWORD *)(v45 + 16LL * v46);
          v49 = *v48;
          if ( v41 != *v48 )
          {
            v70 = 1;
            v71 = 0;
            while ( v49 != -8 )
            {
              if ( !v71 && v49 == -16 )
                v71 = v48;
              v46 = v44 & (v70 + v46);
              v48 = (_QWORD *)(v45 + 16LL * v46);
              v49 = *v48;
              if ( v41 == *v48 )
                goto LABEL_48;
              ++v70;
            }
            if ( v71 )
              v48 = v71;
          }
          goto LABEL_48;
        }
        v37 = (v42 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
        v38 = (_QWORD *)(*((_QWORD *)a1 + 2) + 16LL * v37);
        v39 = *v38;
        if ( v41 != *v38 )
          break;
LABEL_41:
        v35 = *(_QWORD *)(v35 + 8);
        if ( v36 == v35 )
          goto LABEL_51;
      }
      v73 = *((_QWORD *)a1 + 2);
      v61 = 1;
      v48 = 0;
      while ( v39 != -8 )
      {
        if ( v39 != -16 || v48 )
          v38 = v48;
        v37 = (v42 - 1) & (v61 + v37);
        v39 = *(_QWORD *)(v73 + 16LL * v37);
        if ( v41 == v39 )
          goto LABEL_41;
        ++v61;
        v48 = v38;
        v38 = (_QWORD *)(v73 + 16LL * v37);
      }
      if ( !v48 )
        v48 = v38;
      v62 = a1[6];
      ++*((_QWORD *)a1 + 1);
      v47 = v62 + 1;
      if ( 4 * (v62 + 1) >= 3 * v42 )
        goto LABEL_46;
      if ( v42 - a1[7] - v47 <= v42 >> 3 )
      {
        v74 = v36;
        sub_1541C50(v75, v42);
        v63 = a1[8];
        if ( !v63 )
        {
LABEL_135:
          ++a1[6];
          BUG();
        }
        v64 = v63 - 1;
        v65 = *((_QWORD *)a1 + 2);
        v66 = 1;
        v67 = v64 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
        v36 = v74;
        v47 = a1[6] + 1;
        v68 = 0;
        v48 = (_QWORD *)(v65 + 16LL * v67);
        v69 = *v48;
        if ( v41 != *v48 )
        {
          while ( v69 != -8 )
          {
            if ( !v68 && v69 == -16 )
              v68 = v48;
            v67 = v64 & (v66 + v67);
            v48 = (_QWORD *)(v65 + 16LL * v67);
            v69 = *v48;
            if ( v41 == *v48 )
              goto LABEL_48;
            ++v66;
          }
          if ( v68 )
            v48 = v68;
        }
      }
LABEL_48:
      a1[6] = v47;
      if ( *v48 != -8 )
        --a1[7];
      *v48 = v41;
      *((_DWORD *)v48 + 2) = v40;
      v35 = *(_QWORD *)(v35 + 8);
    }
    while ( v36 != v35 );
LABEL_51:
    v32 = v94;
    v34 = *(_QWORD **)(v94 - 24);
LABEL_52:
    while ( 2 )
    {
      if ( !*(_BYTE *)(v32 - 8) )
      {
        v50 = (__int64 *)v34[3];
        *(_BYTE *)(v32 - 8) = 1;
        *(_QWORD *)(v32 - 16) = v50;
        goto LABEL_56;
      }
      while ( 1 )
      {
        v50 = *(__int64 **)(v32 - 16);
LABEL_56:
        if ( v50 == (__int64 *)v34[4] )
          break;
        *(_QWORD *)(v32 - 16) = v50 + 1;
        v52 = *v50;
        v53 = v87;
        if ( v88 == v87 )
        {
          v54 = &v87[v90];
          if ( v87 != v54 )
          {
            v55 = 0;
            while ( v52 != *v53 )
            {
              if ( *v53 == -2 )
              {
                v55 = v53;
                if ( v53 + 1 == v54 )
                  goto LABEL_64;
                ++v53;
              }
              else if ( v54 == ++v53 )
              {
                if ( !v55 )
                  goto LABEL_94;
LABEL_64:
                *v55 = v52;
                --v91;
                ++v86;
                goto LABEL_65;
              }
            }
            continue;
          }
LABEL_94:
          if ( v90 < v89 )
          {
            ++v90;
            *v54 = v52;
            ++v86;
LABEL_65:
            v77 = v52;
            LOBYTE(v79) = 0;
            sub_13B8390(&v93, (__int64)&v77);
            v33 = v93;
            v32 = v94;
            goto LABEL_66;
          }
        }
        sub_16CCBA0((__int64)&v86, v52);
        if ( v51 )
          goto LABEL_65;
      }
      v94 -= 24LL;
      v33 = v93;
      v32 = v94;
      if ( v94 != v93 )
      {
        v34 = *(_QWORD **)(v94 - 24);
        continue;
      }
      break;
    }
LABEL_66:
    v27 = v100;
  }
  while ( v32 - v33 != v101 - v100 );
LABEL_67:
  if ( v32 != v33 )
  {
    v56 = v27;
    while ( *(_QWORD *)v33 == *(_QWORD *)v56 )
    {
      v57 = *(_BYTE *)(v33 + 16);
      v58 = *(_BYTE *)(v56 + 16);
      if ( v57 && v58 )
        v59 = *(_QWORD *)(v33 + 8) == *(_QWORD *)(v56 + 8);
      else
        v59 = v58 == v57;
      if ( !v59 )
        break;
      v33 += 24LL;
      v56 += 24LL;
      if ( v33 == v32 )
        goto LABEL_75;
    }
    goto LABEL_38;
  }
LABEL_75:
  if ( v27 )
    j_j___libc_free_0(v27, v102 - v27);
  if ( v98 != v97 )
    _libc_free(v98);
  if ( v93 )
    j_j___libc_free_0(v93, v95 - v93);
  if ( v88 != v87 )
    _libc_free((unsigned __int64)v88);
  if ( v112 )
    j_j___libc_free_0(v112, v114 - v112);
  if ( v110 != v109[1] )
    _libc_free(v110);
  if ( v106 )
    j_j___libc_free_0(v106, v108 - (_QWORD)v106);
  if ( v104 != v103[1] )
    _libc_free(v104);
  return a1;
}
