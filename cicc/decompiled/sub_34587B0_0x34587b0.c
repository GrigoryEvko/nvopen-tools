// Function: sub_34587B0
// Address: 0x34587b0
//
__int64 __fastcall sub_34587B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        __m128i a7,
        __int128 a8,
        unsigned int a9)
{
  __int64 v9; // r13
  __int64 *v11; // rdi
  __int64 v12; // r13
  unsigned int v13; // eax
  unsigned int v14; // edx
  unsigned int v15; // ebx
  __int128 v16; // rax
  __int64 v17; // r9
  __int128 v18; // rax
  __int128 v19; // rax
  unsigned __int8 *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r15
  unsigned __int8 *v23; // r14
  __int64 v24; // r9
  __int128 v25; // rax
  __int64 v26; // r9
  __int128 v27; // rax
  __int64 v28; // r9
  __int64 v29; // rdx
  __int64 v30; // r14
  __int64 (__fastcall *v31)(__int64, __int64, unsigned int); // rax
  _DWORD *v32; // rax
  unsigned __int16 v33; // r8
  int v34; // eax
  unsigned __int8 *v35; // rax
  __int64 v36; // r8
  __int64 v37; // r9
  unsigned int v38; // edx
  char *v39; // r14
  size_t v40; // r12
  __int64 *v41; // rax
  __int64 **v42; // rax
  unsigned __int64 v43; // r12
  unsigned __int16 v44; // bx
  __int64 (__fastcall *v45)(__int64, __int64, unsigned int); // rax
  int v46; // edx
  unsigned __int16 v47; // ax
  _QWORD *v48; // rax
  __int64 v49; // rdx
  unsigned __int8 *v50; // rax
  __int64 v51; // rdx
  __int128 v52; // rax
  __int64 v53; // r14
  __int64 (__fastcall *v54)(__int64, __int64, __int64, _QWORD, __int64); // rbx
  __int64 v55; // rax
  int v56; // eax
  __int64 v57; // rdx
  __int64 v58; // r14
  __int64 v59; // rdx
  __int128 v60; // rax
  __int64 v61; // r14
  int v62; // edx
  unsigned __int8 *v63; // rax
  __int64 v64; // rdx
  __int64 v65; // r8
  unsigned __int8 *v66; // r10
  __int64 v67; // r11
  __int64 v68; // r12
  __int64 v69; // rbx
  __int16 v70; // ax
  __int64 v71; // r9
  unsigned int v72; // esi
  bool v73; // al
  unsigned int v74; // r13d
  unsigned __int64 v75; // rdx
  char v76; // r14
  unsigned __int64 v77; // rdx
  unsigned __int64 v78; // rdx
  unsigned int v79; // eax
  __int128 v80; // [rsp-10h] [rbp-180h]
  __int128 v81; // [rsp+0h] [rbp-170h]
  __int64 v82; // [rsp+10h] [rbp-160h]
  unsigned __int64 v83; // [rsp+18h] [rbp-158h]
  __int128 n; // [rsp+20h] [rbp-150h]
  __int128 v85; // [rsp+50h] [rbp-120h]
  unsigned __int8 *v86; // [rsp+50h] [rbp-120h]
  __int64 v87; // [rsp+50h] [rbp-120h]
  __int128 v88; // [rsp+50h] [rbp-120h]
  __int128 v90; // [rsp+60h] [rbp-110h]
  unsigned __int8 *v91; // [rsp+60h] [rbp-110h]
  __int64 v92; // [rsp+68h] [rbp-108h]
  __int64 v93; // [rsp+70h] [rbp-100h]
  unsigned __int64 v97; // [rsp+A0h] [rbp-D0h] BYREF
  unsigned int v98; // [rsp+A8h] [rbp-C8h]
  unsigned __int64 v99; // [rsp+B0h] [rbp-C0h] BYREF
  unsigned int v100; // [rsp+B8h] [rbp-B8h]
  __int128 v101; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v102; // [rsp+D0h] [rbp-A0h]
  unsigned __int64 v103; // [rsp+E0h] [rbp-90h] BYREF
  __int64 v104; // [rsp+E8h] [rbp-88h]
  __int64 v105; // [rsp+F0h] [rbp-80h]
  __int64 v106; // [rsp+F8h] [rbp-78h]
  void *v107; // [rsp+100h] [rbp-70h] BYREF
  size_t v108; // [rsp+108h] [rbp-68h]
  __int64 v109; // [rsp+110h] [rbp-60h]
  _BYTE v110[88]; // [rsp+118h] [rbp-58h] BYREF

  if ( ((a9 - 32) & 0xFFFFFFDF) != 0 )
    return 0;
  v11 = *(__int64 **)(a3 + 40);
  v12 = a3;
  if ( a9 == 32 )
  {
    v98 = 32;
    v97 = 125613361;
    v93 = sub_2E79000(v11);
    sub_2EAC2B0((__int64)&v101, *(_QWORD *)(v12 + 40));
    goto LABEL_6;
  }
  v98 = 64;
  v97 = 0x218A392CD3D5DBFLL;
  v93 = sub_2E79000(v11);
  sub_2EAC2B0((__int64)&v101, *(_QWORD *)(v12 + 40));
  v13 = -1;
  if ( a9 )
  {
LABEL_6:
    _BitScanReverse(&v14, a9);
    v13 = 31 - (v14 ^ 0x1F);
  }
  v15 = a9 - v13;
  *(_QWORD *)&v16 = sub_3400BD0(v12, 0, a4, a5, a6, 0, a7, 0);
  *(_QWORD *)&v18 = sub_3406EB0((_QWORD *)v12, 0x39u, a4, a5, a6, v17, v16, a8);
  n = v18;
  *(_QWORD *)&v19 = sub_3400BD0(v12, v15, a4, a5, a6, 0, a7, 0);
  v85 = v19;
  v20 = sub_34007B0(v12, (__int64)&v97, a4, a5, a6, 0, a7, 0);
  v22 = v21;
  v23 = v20;
  *(_QWORD *)&v25 = sub_3406EB0((_QWORD *)v12, 0xBAu, a4, a5, a6, v24, a8, n);
  *((_QWORD *)&v81 + 1) = v22;
  *(_QWORD *)&v81 = v23;
  *(_QWORD *)&v27 = sub_3406EB0((_QWORD *)v12, 0x3Au, a4, a5, a6, v26, v25, v81);
  v86 = sub_3406EB0((_QWORD *)v12, 0xC0u, a4, a5, a6, v28, v27, v85);
  v30 = v29;
  v31 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)a1 + 32LL);
  if ( v31 == sub_2D42F30 )
  {
    v32 = sub_AE2980(v93, 0);
    v33 = 2;
    v34 = v32[1];
    if ( v34 != 1 )
    {
      v33 = 3;
      if ( v34 != 2 )
      {
        v33 = 4;
        if ( v34 != 4 )
        {
          v33 = 5;
          if ( v34 != 8 )
          {
            v33 = 6;
            if ( v34 != 16 )
            {
              v33 = 7;
              if ( v34 != 32 )
              {
                v33 = 8;
                if ( v34 != 64 )
                  v33 = 9 * (v34 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v33 = v31(a1, v93, 0);
  }
  v35 = sub_33FB160(v12, (__int64)v86, v30, a4, v33, 0, a7);
  v108 = 0;
  v87 = (__int64)v35;
  v109 = 40;
  v83 = v38 | v30 & 0xFFFFFFFF00000000LL;
  v107 = v110;
  if ( a9 > 0x28 )
  {
    sub_C8D290((__int64)&v107, v110, a9, 1u, v36, v37);
    memset(v107, 0, a9);
    v108 = a9;
  }
  else
  {
    if ( !a9 )
    {
      v39 = v110;
      v40 = 0;
      goto LABEL_19;
    }
    memset(v110, 0, a9);
    v108 = a9;
  }
  v82 = v12;
  v74 = (a9 - 32) & 0xFFFFFFDF;
  do
  {
    v79 = v98;
    v100 = v98;
    if ( v98 <= 0x40 )
    {
      v99 = v97;
LABEL_43:
      v75 = 0;
      if ( v74 != v79 )
        v75 = v99 << v74;
      LODWORD(v104) = v79;
      v76 = v74;
      v77 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v79) & v75;
      if ( !v79 )
        v77 = 0;
      v99 = v77;
LABEL_48:
      v103 = v99;
LABEL_49:
      if ( v15 == v79 )
        v103 = 0;
      else
        v103 >>= v15;
      goto LABEL_51;
    }
    sub_C43780((__int64)&v99, (const void **)&v97);
    v79 = v100;
    if ( v100 <= 0x40 )
      goto LABEL_43;
    v76 = v74;
    sub_C47690((__int64 *)&v99, v74);
    v79 = v100;
    LODWORD(v104) = v100;
    if ( v100 <= 0x40 )
      goto LABEL_48;
    sub_C43780((__int64)&v103, (const void **)&v99);
    v79 = v104;
    if ( (unsigned int)v104 <= 0x40 )
      goto LABEL_49;
    sub_C482E0((__int64)&v103, v15);
    if ( (unsigned int)v104 > 0x40 )
    {
      v78 = *(_QWORD *)v103;
      goto LABEL_52;
    }
LABEL_51:
    v78 = v103;
LABEL_52:
    *((_BYTE *)v107 + v78) = v76;
    if ( (unsigned int)v104 > 0x40 && v103 )
      j_j___libc_free_0_0(v103);
    if ( v100 > 0x40 && v99 )
      j_j___libc_free_0_0(v99);
    ++v74;
  }
  while ( a9 > v74 );
  v12 = v82;
  v39 = (char *)v107;
  v40 = v108;
LABEL_19:
  v41 = (__int64 *)sub_BCD140(*(_QWORD **)(v12 + 64), 8u);
  v42 = (__int64 **)sub_BCD420(v41, v40);
  v43 = sub_AC9630(v39, v40, v42);
  LOBYTE(v44) = sub_AE5260(v93, *(_QWORD *)(v43 + 8));
  HIBYTE(v44) = 1;
  v45 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)a1 + 32LL);
  if ( v45 == sub_2D42F30 )
  {
    v46 = sub_AE2980(v93, 0)[1];
    v47 = 2;
    if ( v46 != 1 )
    {
      v47 = 3;
      if ( v46 != 2 )
      {
        v47 = 4;
        if ( v46 != 4 )
        {
          v47 = 5;
          if ( v46 != 8 )
          {
            v47 = 6;
            if ( v46 != 16 )
            {
              v47 = 7;
              if ( v46 != 32 )
              {
                v47 = 8;
                if ( v46 != 64 )
                  v47 = 9 * (v46 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v47 = v45(a1, v93, 0);
  }
  v48 = sub_33EE5B0((_QWORD *)v12, v43, v47, 0, v44, 0, 0, 0);
  v105 = 0;
  v103 = 0;
  v104 = 0;
  v106 = 0;
  v50 = sub_34092D0((_QWORD *)v12, (__int64)v48, v49, v87, v83, a4, a7, 0);
  *(_QWORD *)&v52 = sub_33F1DB0(
                      (__int64 *)v12,
                      3,
                      a4,
                      a5,
                      a6,
                      0,
                      (unsigned __int64)(v12 + 288),
                      (__int64)v50,
                      v51,
                      v101,
                      v102,
                      5,
                      0,
                      0,
                      (__int64)&v103);
  v88 = v52;
  if ( *(_DWORD *)(a2 + 24) == 203 )
  {
    v9 = v52;
  }
  else
  {
    v53 = *(_QWORD *)(v12 + 64);
    v54 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)a1 + 528LL);
    v55 = sub_2E79000(*(__int64 **)(v12 + 40));
    v56 = v54(a1, v55, v53, a5, a6);
    v58 = v57;
    LODWORD(v54) = v56;
    *(_QWORD *)&v90 = sub_3400BD0(v12, 0, a4, a5, a6, 0, a7, 0);
    *((_QWORD *)&v90 + 1) = v59;
    *(_QWORD *)&v60 = sub_33ED040((_QWORD *)v12, 0x11u);
    v61 = sub_340F900((_QWORD *)v12, 0xD0u, a4, (unsigned int)v54, v58, *((__int64 *)&v90 + 1), a8, v90, v60);
    LODWORD(v54) = v62;
    v63 = sub_3400BD0(v12, a9, a4, a5, a6, 0, a7, 0);
    v65 = v61;
    v66 = v63;
    v67 = v64;
    v68 = (unsigned int)v54;
    v69 = *(_QWORD *)(v61 + 48) + 16LL * (unsigned int)v54;
    v70 = *(_WORD *)v69;
    v71 = v68;
    v104 = *(_QWORD *)(v69 + 8);
    LOWORD(v103) = v70;
    if ( v70 )
    {
      v72 = ((unsigned __int16)(v70 - 17) < 0xD4u) + 205;
    }
    else
    {
      v91 = v66;
      v92 = v64;
      v73 = sub_30070B0((__int64)&v103);
      v65 = v61;
      v71 = v68;
      v66 = v91;
      v67 = v92;
      v72 = 205 - (!v73 - 1);
    }
    *((_QWORD *)&v80 + 1) = v67;
    *(_QWORD *)&v80 = v66;
    v9 = sub_340EC60((_QWORD *)v12, v72, a4, a5, a6, 0, v65, v71, v80, v88);
  }
  if ( v107 != v110 )
    _libc_free((unsigned __int64)v107);
  if ( v98 > 0x40 && v97 )
    j_j___libc_free_0_0(v97);
  return v9;
}
