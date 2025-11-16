// Function: sub_1965DD0
// Address: 0x1965dd0
//
__int64 __fastcall sub_1965DD0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 **a6,
        __m128 a7,
        __m128i a8,
        __m128i a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15,
        __int64 a16,
        __int64 a17,
        __int64 *a18)
{
  __int64 v18; // r13
  int v19; // ecx
  __int64 v20; // rbx
  __int64 v21; // rax
  int v22; // ecx
  __int64 v23; // rsi
  unsigned int v24; // edx
  __int64 *v25; // rax
  __int64 v26; // rdi
  _QWORD *v27; // r14
  bool v28; // zf
  __int64 v29; // r12
  int v30; // r9d
  char v31; // dl
  __int64 v32; // rax
  __int64 *v33; // r15
  __int64 v34; // rax
  __int64 v35; // rax
  unsigned __int64 v36; // r9
  unsigned __int64 v37; // r8
  _BYTE *v38; // rcx
  int v39; // esi
  _BYTE *v40; // rdx
  __int64 *v41; // rax
  __int64 v42; // rdx
  int v43; // r15d
  __int64 v44; // r12
  __int64 v45; // r14
  _QWORD *v46; // rbx
  _QWORD *v47; // r13
  _QWORD *v48; // rax
  _QWORD *v49; // rax
  __int64 v50; // rdx
  _QWORD *v51; // r15
  __int64 v52; // rdi
  __int64 *v53; // r15
  __int64 v54; // rax
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  unsigned __int64 v57; // r8
  _BYTE *v58; // rcx
  int v59; // esi
  _BYTE *v60; // r9
  __int64 *v61; // rax
  __int64 v62; // rsi
  int v63; // r15d
  bool v64; // r15
  __int64 v65; // rbx
  _QWORD *v66; // rax
  __int64 v67; // r14
  double v68; // xmm4_8
  double v69; // xmm5_8
  unsigned __int8 v70; // bl
  unsigned int v71; // ecx
  __int64 v72; // rdi
  unsigned int v73; // edx
  __int64 *v74; // rax
  __int64 v75; // r10
  __int64 v76; // rax
  _QWORD *v77; // rdx
  int v78; // eax
  int v80; // eax
  int v81; // r8d
  int v82; // esi
  __int64 v83; // rax
  __int64 v86; // [rsp+20h] [rbp-150h]
  char *v87; // [rsp+28h] [rbp-148h]
  unsigned __int64 v91; // [rsp+48h] [rbp-128h]
  __int64 v92; // [rsp+50h] [rbp-120h]
  unsigned __int8 v93; // [rsp+5Eh] [rbp-112h]
  unsigned __int8 v94; // [rsp+5Fh] [rbp-111h]
  char *v95; // [rsp+60h] [rbp-110h]
  __int64 v96; // [rsp+68h] [rbp-108h]
  unsigned __int64 v97; // [rsp+68h] [rbp-108h]
  unsigned __int64 v98; // [rsp+68h] [rbp-108h]
  __int64 v99; // [rsp+70h] [rbp-100h]
  __int64 v100; // [rsp+70h] [rbp-100h]
  __int64 v101; // [rsp+70h] [rbp-100h]
  __int64 v102; // [rsp+78h] [rbp-F8h]
  _QWORD *v103; // [rsp+78h] [rbp-F8h]
  _BYTE *v104; // [rsp+80h] [rbp-F0h] BYREF
  __int64 v105; // [rsp+88h] [rbp-E8h]
  _BYTE v106[32]; // [rsp+90h] [rbp-E0h] BYREF
  char *v107; // [rsp+B0h] [rbp-C0h] BYREF
  int v108; // [rsp+B8h] [rbp-B8h]
  char v109; // [rsp+C0h] [rbp-B0h] BYREF

  v18 = a15;
  sub_1B1AE10(&v107, a1, a15);
  v93 = 0;
  v87 = v107;
  v95 = &v107[8 * v108];
  if ( v107 != v95 )
  {
    while ( 1 )
    {
      v19 = *(_DWORD *)(a3 + 24);
      v20 = **((_QWORD **)v95 - 1);
      v21 = 0;
      v92 = v20;
      if ( v19 )
      {
        v22 = v19 - 1;
        v23 = *(_QWORD *)(a3 + 8);
        v24 = v22 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v25 = (__int64 *)(v23 + 16LL * v24);
        v26 = *v25;
        if ( v20 == *v25 )
        {
LABEL_6:
          v21 = v25[1];
        }
        else
        {
          v80 = 1;
          while ( v26 != -8 )
          {
            v81 = v80 + 1;
            v24 = v22 & (v80 + v24);
            v25 = (__int64 *)(v23 + 16LL * v24);
            v26 = *v25;
            if ( v20 == *v25 )
              goto LABEL_6;
            v80 = v81;
          }
          v21 = 0;
        }
      }
      if ( v18 == v21 )
      {
        v27 = (_QWORD *)(v20 + 40);
        if ( *(_QWORD *)(v20 + 48) != v20 + 40 )
          break;
      }
LABEL_3:
      v95 -= 8;
      if ( v87 == v95 )
      {
        v95 = v107;
        goto LABEL_92;
      }
    }
    while ( 1 )
    {
      v28 = (*v27 & 0xFFFFFFFFFFFFFFF8LL) == 0;
      v91 = *v27 & 0xFFFFFFFFFFFFFFF8LL;
      v29 = v91 - 24;
      v27 = (_QWORD *)v91;
      if ( v28 )
        v29 = 0;
      v94 = sub_1AE9990(v29, a5);
      if ( v94 )
      {
        sub_1AEAA40(v29);
        v27 = *(_QWORD **)(v91 + 8);
        sub_1359860(a16, v29);
        sub_15F20C0((_QWORD *)v29);
        v93 = v94;
        goto LABEL_47;
      }
      v31 = *(_BYTE *)(v29 + 23) & 0x40;
      v32 = 3LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF);
      if ( *(_BYTE *)(v29 + 16) != 56 )
        break;
      if ( v31 )
      {
        v33 = *(__int64 **)(v29 - 8);
        v34 = (__int64)&v33[v32];
      }
      else
      {
        v33 = (__int64 *)(v29 - v32 * 8);
        v34 = v29;
      }
      v35 = v34 - (_QWORD)v33;
      v104 = v106;
      v105 = 0x400000000LL;
      v36 = 0xAAAAAAAAAAAAAAABLL * (v35 >> 3);
      v37 = v36;
      if ( (unsigned __int64)v35 > 0x60 )
      {
        v97 = 0xAAAAAAAAAAAAAAABLL * (v35 >> 3);
        v100 = v35;
        sub_16CD150((__int64)&v104, v106, v97, 8, v36, v36);
        v40 = v104;
        v39 = v105;
        LODWORD(v36) = v97;
        v35 = v100;
        v37 = v97;
        v38 = &v104[8 * (unsigned int)v105];
      }
      else
      {
        v38 = v106;
        v39 = 0;
        v40 = v106;
      }
      if ( v35 > 0 )
      {
        v41 = v33;
        do
        {
          v42 = *v41;
          v38 += 8;
          v41 += 3;
          *((_QWORD *)v38 - 1) = v42;
          --v37;
        }
        while ( v37 );
        v40 = v104;
        v39 = v105;
      }
      LODWORD(v105) = v39 + v36;
      v43 = sub_14A5330(a6, v29, (__int64)v40, (unsigned int)(v39 + v36));
      if ( v104 != v106 )
        _libc_free((unsigned __int64)v104);
      if ( v43 )
      {
LABEL_82:
        v64 = 0;
        goto LABEL_60;
      }
      v99 = *(_QWORD *)(v29 + 40);
      if ( *(_QWORD *)(v29 + 8) )
      {
        v86 = v29;
        v44 = *(_QWORD *)(v29 + 8);
        v45 = v18;
        v96 = v18 + 56;
        v46 = *(_QWORD **)(v18 + 72);
        while ( 1 )
        {
          v49 = sub_1648700(v44);
          v50 = v49[5];
          v51 = v49;
          v48 = *(_QWORD **)(v45 + 64);
          if ( v46 == v48 )
          {
            v52 = *(unsigned int *)(v45 + 84);
            v47 = &v46[v52];
            if ( v46 == v47 )
            {
              v77 = v46;
            }
            else
            {
              do
              {
                if ( v50 == *v48 )
                  break;
                ++v48;
              }
              while ( v47 != v48 );
              v77 = &v46[v52];
            }
            goto LABEL_41;
          }
          v102 = v50;
          v47 = &v46[*(unsigned int *)(v45 + 80)];
          v48 = sub_16CC9F0(v96, v50);
          if ( v102 == *v48 )
            break;
          v46 = *(_QWORD **)(v45 + 72);
          if ( v46 == *(_QWORD **)(v45 + 64) )
          {
            v48 = &v46[*(unsigned int *)(v45 + 84)];
            v77 = v48;
            goto LABEL_41;
          }
          v48 = &v46[*(unsigned int *)(v45 + 80)];
LABEL_29:
          if ( v47 != v48 && (v99 != v51[5] || (unsigned __int8)(*((_BYTE *)v51 + 16) - 54) > 1u) )
          {
            v18 = v45;
            v29 = v86;
            v27 = (_QWORD *)v91;
            goto LABEL_82;
          }
          v44 = *(_QWORD *)(v44 + 8);
          if ( !v44 )
          {
            v18 = v45;
            v29 = v86;
            v64 = 1;
            v27 = (_QWORD *)v91;
            goto LABEL_60;
          }
        }
        v46 = *(_QWORD **)(v45 + 72);
        if ( v46 == *(_QWORD **)(v45 + 64) )
          v77 = &v46[*(unsigned int *)(v45 + 84)];
        else
          v77 = &v46[*(unsigned int *)(v45 + 80)];
LABEL_41:
        while ( v77 != v48 && *v48 >= 0xFFFFFFFFFFFFFFFELL )
          ++v48;
        goto LABEL_29;
      }
LABEL_70:
      if ( (unsigned __int8)sub_1960590(v29, a2, a4, v18, a16, (char *)a17, a18) )
      {
        v70 = sub_19636E0(v29, a3, a4, v18, a17, a18, a7, a8, a9, a10, v68, v69, a13, a14);
        if ( v70 )
        {
          v93 = v94;
          if ( !v94 )
          {
            v27 = *(_QWORD **)(v91 + 8);
            sub_1359860(a16, v29);
            sub_15F20C0((_QWORD *)v29);
            v93 = v70;
          }
        }
      }
LABEL_47:
      if ( *(_QWORD **)(v92 + 48) == v27 )
        goto LABEL_3;
    }
    if ( v31 )
    {
      v53 = *(__int64 **)(v29 - 8);
      v54 = (__int64)&v53[v32];
    }
    else
    {
      v53 = (__int64 *)(v29 - v32 * 8);
      v54 = v29;
    }
    v55 = v54 - (_QWORD)v53;
    v105 = 0x400000000LL;
    v104 = v106;
    v56 = 0xAAAAAAAAAAAAAAABLL * (v55 >> 3);
    v57 = v56;
    if ( (unsigned __int64)v55 > 0x60 )
    {
      v98 = 0xAAAAAAAAAAAAAAABLL * (v55 >> 3);
      v101 = v55;
      sub_16CD150((__int64)&v104, v106, v56, 8, v56, v30);
      v60 = v104;
      v59 = v105;
      LODWORD(v56) = v98;
      v55 = v101;
      v57 = v98;
      v58 = &v104[8 * (unsigned int)v105];
    }
    else
    {
      v58 = v106;
      v59 = 0;
      v60 = v106;
    }
    if ( v55 > 0 )
    {
      v61 = v53;
      do
      {
        v62 = *v61;
        v58 += 8;
        v61 += 3;
        *((_QWORD *)v58 - 1) = v62;
        --v57;
      }
      while ( v57 );
      v60 = v104;
      v59 = v105;
    }
    LODWORD(v105) = v59 + v56;
    v63 = sub_14A5330(a6, v29, (__int64)v60, (unsigned int)(v59 + v56));
    if ( v104 != v106 )
      _libc_free((unsigned __int64)v104);
    v64 = v63 == 0;
LABEL_60:
    v65 = *(_QWORD *)(v29 + 8);
    if ( !v65 )
      goto LABEL_70;
    v103 = v27;
    while ( 1 )
    {
      v66 = sub_1648700(v65);
      v67 = v66[5];
      if ( *((_BYTE *)v66 + 16) != 77 )
        goto LABEL_65;
      if ( *(_BYTE *)(sub_157EBA0(v66[5]) + 16) == 34 )
        goto LABEL_80;
      if ( *(_BYTE *)(v29 + 16) != 78 || !*(_DWORD *)(a17 + 24) )
        goto LABEL_65;
      v71 = *(_DWORD *)(a17 + 32);
      v72 = *(_QWORD *)(a17 + 16);
      if ( !v71 )
        goto LABEL_90;
      v73 = (v71 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
      v74 = (__int64 *)(v72 + 16LL * v73);
      v75 = *v74;
      if ( v67 != *v74 )
        break;
LABEL_77:
      v76 = v74[1];
      if ( (v76 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v76 & 4) != 0 && *(_DWORD *)((v76 & 0xFFFFFFFFFFFFFFF8LL) + 8) != 1 )
      {
LABEL_80:
        v27 = v103;
        goto LABEL_47;
      }
LABEL_65:
      if ( sub_1377F70(v18 + 56, v67) )
      {
        if ( !v64 )
          goto LABEL_80;
        v94 = v64;
      }
      v65 = *(_QWORD *)(v65 + 8);
      if ( !v65 )
      {
        v27 = v103;
        goto LABEL_70;
      }
    }
    v78 = 1;
    while ( v75 != -8 )
    {
      v82 = v78 + 1;
      v83 = (v71 - 1) & (v73 + v78);
      v73 = v83;
      v74 = (__int64 *)(v72 + 16 * v83);
      v75 = *v74;
      if ( v67 == *v74 )
        goto LABEL_77;
      v78 = v82;
    }
LABEL_90:
    v74 = (__int64 *)(v72 + 16LL * v71);
    goto LABEL_77;
  }
LABEL_92:
  if ( v95 != &v109 )
    _libc_free((unsigned __int64)v95);
  return v93;
}
