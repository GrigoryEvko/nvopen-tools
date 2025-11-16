// Function: sub_2183E30
// Address: 0x2183e30
//
__int64 __fastcall sub_2183E30(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v8; // r8
  int v9; // r9d
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rbx
  int v13; // r13d
  __int64 v14; // r12
  __int64 v15; // rdi
  int v16; // eax
  unsigned int v17; // eax
  __int64 v18; // rcx
  __int64 v19; // r13
  __int64 v20; // rdx
  __int64 v21; // rbx
  __int64 v22; // r12
  __int64 v23; // rax
  int v24; // esi
  __int64 v25; // rax
  unsigned int *v26; // rdi
  __int64 v27; // rax
  int v28; // r11d
  _QWORD *v29; // r10
  __int64 v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rcx
  int v35; // r9d
  __int64 v36; // r14
  __int64 *v37; // r13
  unsigned int v38; // r15d
  unsigned int v39; // r12d
  __int64 v40; // rax
  int v41; // ecx
  __int64 v42; // rsi
  int v43; // ecx
  unsigned int v44; // edx
  __int64 *v45; // rax
  __int64 v46; // r9
  unsigned int v47; // edx
  __int64 v48; // rsi
  int v49; // r12d
  __int64 v50; // rcx
  __int64 v51; // r9
  __int64 *v52; // r15
  char v53; // al
  int v54; // edi
  int v55; // r10d
  __int64 *v57; // r12
  __int64 *v58; // rbx
  unsigned int v59; // edx
  __int64 v60; // rax
  unsigned int v61; // r14d
  __int64 v62; // r15
  __int64 v63; // r13
  bool v64; // dl
  __int64 v65; // rax
  __int64 v66; // rcx
  int v67; // eax
  __int64 v68; // rsi
  int v69; // eax
  unsigned int v70; // edi
  __int64 *v71; // rdx
  __int64 v72; // r10
  __int64 v73; // rax
  int v74; // r9d
  __int64 *v75; // rsi
  __int64 v76; // rdi
  __int64 v77; // r10
  unsigned int v78; // edi
  __int64 *v79; // rsi
  __int64 v80; // r9
  __int64 v81; // rdi
  bool v82; // zf
  char v83; // di
  int v84; // edx
  __int64 v85; // rdx
  __int64 v86; // rcx
  int v87; // r8d
  int v88; // r9d
  __int64 *v89; // rax
  unsigned int v90; // esi
  int v91; // ecx
  int v92; // edx
  __int64 v93; // rdx
  int v94; // esi
  int v95; // esi
  int v96; // r8d
  int v97; // eax
  int v98; // edi
  int v99; // r8d
  __int64 v100; // rsi
  int v101; // r8d
  unsigned int v103; // [rsp+8h] [rbp-F8h]
  unsigned int v105; // [rsp+14h] [rbp-ECh]
  unsigned int v107; // [rsp+30h] [rbp-D0h]
  char v108; // [rsp+38h] [rbp-C8h]
  __int64 v110; // [rsp+48h] [rbp-B8h] BYREF
  __int64 v111; // [rsp+50h] [rbp-B0h] BYREF
  __int64 *v112; // [rsp+58h] [rbp-A8h] BYREF
  __int64 v113; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v114; // [rsp+68h] [rbp-98h]
  __int64 v115; // [rsp+70h] [rbp-90h]
  __int64 v116; // [rsp+78h] [rbp-88h]
  _QWORD *v117; // [rsp+80h] [rbp-80h] BYREF
  __int64 *v118; // [rsp+88h] [rbp-78h]
  __int64 v119; // [rsp+90h] [rbp-70h]
  __int64 v120; // [rsp+98h] [rbp-68h]
  _BYTE *v121; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v122; // [rsp+A8h] [rbp-58h]
  _BYTE v123[80]; // [rsp+B0h] [rbp-50h] BYREF

  v121 = v123;
  v122 = 0x400000000LL;
  v110 = a3;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  sub_217E700((__int64)&v121, &v110, a3, a4, a5, a6);
  v10 = v110;
  v11 = *(unsigned int *)(v110 + 40);
  if ( (_DWORD)v11 )
  {
    v12 = 0;
    v13 = 0;
    v14 = 40 * v11;
    while ( 1 )
    {
      v15 = v12 + *(_QWORD *)(v10 + 32);
      if ( *(_BYTE *)v15 || (*(_BYTE *)(v15 + 3) & 0x10) == 0 )
      {
        v12 += 40;
        if ( v12 == v14 )
          goto LABEL_8;
      }
      else
      {
        v12 += 40;
        LODWORD(v117) = *(_DWORD *)(v15 + 8);
        v13 += sub_1E49390(a1 + 352, (int *)&v117)[1];
        if ( v12 == v14 )
        {
LABEL_8:
          v16 = 1;
          if ( v13 )
            v16 = v13;
          v105 = v16;
          goto LABEL_11;
        }
      }
      v10 = v110;
    }
  }
  v105 = 1;
LABEL_11:
  v17 = v122;
  if ( !(_DWORD)v122 )
    goto LABEL_33;
  do
  {
    v18 = v17--;
    v19 = *(_QWORD *)&v121[8 * v18 - 8];
    LODWORD(v122) = v17;
    v20 = *(unsigned int *)(v19 + 40);
    if ( !(_DWORD)v20 )
      continue;
    v21 = 0;
    v22 = 40 * v20;
    do
    {
      while ( 1 )
      {
        v23 = v21 + *(_QWORD *)(v19 + 32);
        if ( *(_BYTE *)v23 || (*(_BYTE *)(v23 + 3) & 0x10) != 0 )
          goto LABEL_14;
        v24 = *(_DWORD *)(v23 + 8);
        v25 = *(unsigned int *)(a4 + 24);
        if ( (_DWORD)v25 )
        {
          v9 = v25 - 1;
          v8 = *(_QWORD *)(a4 + 8);
          v20 = ((_DWORD)v25 - 1) & (unsigned int)(37 * v24);
          v26 = (unsigned int *)(v8 + 4 * v20);
          v18 = *v26;
          if ( v24 == (_DWORD)v18 )
          {
LABEL_19:
            if ( v26 != (unsigned int *)(v8 + 4 * v25) )
              goto LABEL_14;
          }
          else
          {
            v54 = 1;
            while ( (_DWORD)v18 != -1 )
            {
              v55 = v54 + 1;
              v20 = v9 & (unsigned int)(v54 + v20);
              v26 = (unsigned int *)(v8 + 4LL * (unsigned int)v20);
              v18 = *v26;
              if ( v24 == (_DWORD)v18 )
                goto LABEL_19;
              v54 = v55;
            }
          }
        }
        v27 = sub_217E810(a1, v24, v20, v18, v8, v9);
        v112 = (__int64 *)v27;
        if ( !v27 )
        {
          v39 = 0;
          goto LABEL_64;
        }
        if ( !(_DWORD)v116 )
        {
          ++v113;
LABEL_68:
          sub_1E22DE0((__int64)&v113, 2 * v116);
          goto LABEL_69;
        }
        v9 = v116 - 1;
        LODWORD(v8) = v114;
        v28 = 1;
        v29 = 0;
        v20 = ((_DWORD)v116 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v18 = v114 + 8 * v20;
        v30 = *(_QWORD *)v18;
        if ( v27 != *(_QWORD *)v18 )
          break;
LABEL_14:
        v21 += 40;
        if ( v22 == v21 )
          goto LABEL_31;
      }
      while ( v30 != -8 )
      {
        if ( v29 || v30 != -16 )
          v18 = (__int64)v29;
        v20 = v9 & (unsigned int)(v28 + v20);
        v30 = *(_QWORD *)(v114 + 8LL * (unsigned int)v20);
        if ( v27 == v30 )
          goto LABEL_14;
        ++v28;
        v29 = (_QWORD *)v18;
        v18 = v114 + 8LL * (unsigned int)v20;
      }
      if ( !v29 )
        v29 = (_QWORD *)v18;
      ++v113;
      v31 = (unsigned int)(v115 + 1);
      if ( 4 * (int)v31 >= (unsigned int)(3 * v116) )
        goto LABEL_68;
      v32 = (unsigned int)(v116 - HIDWORD(v115) - v31);
      if ( (unsigned int)v32 > (unsigned int)v116 >> 3 )
        goto LABEL_28;
      sub_1E22DE0((__int64)&v113, v116);
LABEL_69:
      sub_1E1F3B0((__int64)&v113, (__int64 *)&v112, &v117);
      v29 = v117;
      v27 = (__int64)v112;
      v31 = (unsigned int)(v115 + 1);
LABEL_28:
      LODWORD(v115) = v31;
      if ( *v29 != -8 )
        --HIDWORD(v115);
      *v29 = v27;
      v21 += 40;
      sub_217E700((__int64)&v121, &v112, v32, v31, (int)&v112, v9);
      sub_217E700(a5, &v112, v33, v34, (int)&v112, v35);
    }
    while ( v22 != v21 );
LABEL_31:
    v17 = v122;
  }
  while ( v17 );
LABEL_33:
  v36 = *(_QWORD *)(v110 + 24);
  v117 = 0;
  v118 = 0;
  v119 = 0;
  v120 = 0;
  sub_2182E00(a1, a2, (__int64)&v117, a6);
  v37 = v118;
  if ( !(_DWORD)v119 || (v57 = &v118[(unsigned int)v120], v118 == v57) )
  {
LABEL_34:
    v107 = 0;
    v38 = 0;
    v108 = 0;
    goto LABEL_35;
  }
  v58 = v118;
  while ( *v58 == -8 || *v58 == -16 )
  {
    if ( v57 == ++v58 )
      goto LABEL_34;
  }
  v108 = 0;
  if ( v58 != v57 )
  {
    v107 = 0;
    v59 = ((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4);
    v60 = v36;
    v61 = 0;
    v103 = v59;
    v62 = v60;
    while ( 1 )
    {
      v63 = *v58;
      if ( *v58 == v62 )
        goto LABEL_85;
      ++v107;
      v64 = sub_1DD6970(v62, *v58);
      v65 = *(_QWORD *)(a1 + 272);
      v66 = *(_QWORD *)(v65 + 240);
      v67 = *(_DWORD *)(v65 + 256);
      if ( v64 )
      {
        if ( !v67 )
          goto LABEL_102;
        v69 = v67 - 1;
        v74 = v69 & v103;
        v75 = (__int64 *)(v66 + 16LL * (v69 & v103));
        v76 = *v75;
        if ( v62 == *v75 )
        {
LABEL_94:
          v77 = v75[1];
        }
        else
        {
          v95 = 1;
          while ( v76 != -8 )
          {
            v99 = v95 + 1;
            v100 = v69 & (unsigned int)(v74 + v95);
            v74 = v100;
            v75 = (__int64 *)(v66 + 16 * v100);
            v76 = *v75;
            if ( v62 == *v75 )
              goto LABEL_94;
            v95 = v99;
          }
          v77 = 0;
        }
        v78 = v69 & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
        v79 = (__int64 *)(v66 + 16LL * v78);
        v80 = *v79;
        if ( v63 == *v79 )
        {
LABEL_96:
          v81 = v79[1];
        }
        else
        {
          v94 = 1;
          while ( v80 != -8 )
          {
            v101 = v94 + 1;
            v78 = v69 & (v94 + v78);
            v79 = (__int64 *)(v66 + 16LL * v78);
            v80 = *v79;
            if ( v63 == *v79 )
              goto LABEL_96;
            v94 = v101;
          }
          v81 = 0;
        }
        v82 = v81 == v77;
        v83 = v108;
        v68 = *v58;
        if ( !v82 )
          v83 = v64;
        v108 = v83;
      }
      else
      {
        v108 = 1;
        v68 = *v58;
        if ( !v67 )
          goto LABEL_102;
        v69 = v67 - 1;
      }
      v70 = v69 & (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4));
      v71 = (__int64 *)(v66 + 16LL * v70);
      v72 = *v71;
      if ( v68 == *v71 )
      {
LABEL_83:
        if ( v71[1] )
        {
          v61 += dword_4FD34A0;
          goto LABEL_85;
        }
      }
      else
      {
        v84 = 1;
        while ( v72 != -8 )
        {
          v96 = v84 + 1;
          v70 = v69 & (v84 + v70);
          v71 = (__int64 *)(v66 + 16LL * v70);
          v72 = *v71;
          if ( *v71 == v68 )
            goto LABEL_83;
          v84 = v96;
        }
      }
LABEL_102:
      ++v61;
LABEL_85:
      if ( ++v58 != v57 )
      {
        while ( *v58 == -16 || *v58 == -8 )
        {
          if ( v57 == ++v58 )
            goto LABEL_89;
        }
        if ( v57 != v58 )
          continue;
      }
LABEL_89:
      v73 = v62;
      v37 = v118;
      v38 = v61;
      v36 = v73;
      goto LABEL_35;
    }
  }
  v107 = 0;
  v38 = 0;
LABEL_35:
  if ( (dword_4FD3820 & 8) == 0 || v108 || (v39 = 0, (unsigned int)v119 <= 1) )
  {
    v40 = *(_QWORD *)(a1 + 272);
    v41 = *(_DWORD *)(v40 + 256);
    if ( v41 )
    {
      v42 = *(_QWORD *)(v40 + 240);
      v43 = v41 - 1;
      v44 = v43 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
      v45 = (__int64 *)(v42 + 16LL * v44);
      v46 = *v45;
      if ( *v45 == v36 )
      {
LABEL_40:
        if ( v45[1] )
          v38 /= (unsigned int)dword_4FD34A0;
      }
      else
      {
        v97 = 1;
        while ( v46 != -8 )
        {
          v98 = v97 + 1;
          v44 = v43 & (v97 + v44);
          v45 = (__int64 *)(v42 + 16LL * v44);
          v46 = *v45;
          if ( v36 == *v45 )
            goto LABEL_40;
          v97 = v98;
        }
      }
    }
    if ( !v38 )
      v38 = 1;
    if ( dword_4FD3120 <= v105 && v107 <= 1
      || (v39 = 0, v38 <= dword_4FD3660)
      && (v47 = *(_DWORD *)(a5 + 8), dword_4FD3660 - 1 >= v47)
      && v47 <= dword_4FD3580 )
    {
      v48 = v110;
      v49 = sub_217D4A0(*(unsigned __int16 **)(v110 + 16));
      if ( *(_DWORD *)(a5 + 8) )
      {
        v50 = *(_QWORD *)a5;
        do
          v49 += sub_217D4A0(*(unsigned __int16 **)(*(_QWORD *)v50 + 16LL));
        while ( v51 != v50 );
      }
      v39 = v38 * v49;
      v52 = &v37[(unsigned int)v120];
      if ( (_DWORD)v119 && v37 != v52 )
      {
        while ( *v37 == -8 || *v37 == -16 )
        {
          if ( v52 == ++v37 )
            goto LABEL_53;
        }
        if ( v37 != v52 )
        {
          while ( 2 )
          {
            v111 = *v37;
            v88 = sub_217F4B0(a7, &v111, &v112);
            v89 = v112;
            if ( !(_BYTE)v88 )
            {
              v90 = *(_DWORD *)(a7 + 24);
              v91 = *(_DWORD *)(a7 + 16);
              ++*(_QWORD *)a7;
              v92 = v91 + 1;
              if ( 4 * (v91 + 1) >= 3 * v90 )
              {
                v90 *= 2;
              }
              else
              {
                v88 = v90 >> 3;
                if ( v90 - *(_DWORD *)(a7 + 20) - v92 > v90 >> 3 )
                  goto LABEL_113;
              }
              sub_2183150(a7, v90);
              sub_217F4B0(a7, &v111, &v112);
              v89 = v112;
              v92 = *(_DWORD *)(a7 + 16) + 1;
LABEL_113:
              *(_DWORD *)(a7 + 16) = v92;
              if ( *v89 != -8 )
                --*(_DWORD *)(a7 + 20);
              v93 = v111;
              v86 = 0x400000000LL;
              v89[2] = 0x400000000LL;
              *v89 = v93;
              v85 = (__int64)(v89 + 3);
              v89[1] = (__int64)(v89 + 3);
            }
            ++v37;
            sub_217E700((__int64)(v89 + 1), &v110, v85, v86, v87, v88);
            if ( v37 == v52 )
              goto LABEL_120;
            while ( *v37 == -16 || *v37 == -8 )
            {
              if ( v52 == ++v37 )
                goto LABEL_120;
            }
            if ( v37 == v52 )
            {
LABEL_120:
              v48 = v110;
              break;
            }
            continue;
          }
        }
      }
LABEL_53:
      if ( v39 > 2 && 3 * dword_4FD34A0 > v39 && v107 == 1 && dword_4FD3120 <= v105 )
        v39 = 2;
      v53 = sub_2183B30(a1, v48);
      v37 = v118;
      if ( v53 )
        v39 = 0;
    }
  }
  j___libc_free_0(v37);
LABEL_64:
  j___libc_free_0(v114);
  if ( v121 != v123 )
    _libc_free((unsigned __int64)v121);
  return v39;
}
