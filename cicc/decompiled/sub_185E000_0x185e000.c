// Function: sub_185E000
// Address: 0x185e000
//
__int64 __fastcall sub_185E000(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 *v18; // r9
  unsigned int v19; // r12d
  double v20; // xmm4_8
  double v21; // xmm5_8
  int v23; // eax
  __int64 *v24; // r15
  __int64 *v25; // r13
  __int128 v26; // rdi
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 *v29; // r15
  __int64 *v30; // r13
  __int64 v31; // rsi
  __int64 v32; // rdi
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rdx
  _BYTE *v36; // rax
  __int64 v37; // rdx
  _QWORD *v38; // rbx
  _QWORD *v39; // r13
  __int64 v40; // rdx
  __int64 v41; // r14
  __int64 v42; // r12
  __int64 v43; // rsi
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 *v46; // rax
  __int64 *v47; // rax
  __int64 *v48; // rdx
  __int64 v49; // r15
  __int64 *v50; // r13
  __int64 v51; // rax
  __int64 *v52; // rax
  __int64 v53; // rax
  __int64 *v54; // rax
  __int64 v55; // rax
  __int64 *v56; // rax
  __int64 *v57; // [rsp+0h] [rbp-A50h]
  __int64 *v58; // [rsp+0h] [rbp-A50h]
  __int64 *v59; // [rsp+0h] [rbp-A50h]
  __int64 v60; // [rsp+8h] [rbp-A48h]
  __int64 v61; // [rsp+8h] [rbp-A48h]
  __int64 v62; // [rsp+8h] [rbp-A48h]
  unsigned __int8 v63; // [rsp+10h] [rbp-A40h]
  unsigned __int8 v64; // [rsp+18h] [rbp-A38h]
  char v65[8]; // [rsp+30h] [rbp-A20h] BYREF
  __int64 v66; // [rsp+38h] [rbp-A18h] BYREF
  _QWORD v67[2]; // [rsp+40h] [rbp-A10h] BYREF
  unsigned __int64 v68[2]; // [rsp+50h] [rbp-A00h] BYREF
  _BYTE v69[256]; // [rsp+60h] [rbp-9F0h] BYREF
  __int64 *v70; // [rsp+160h] [rbp-8F0h] BYREF
  __int64 v71; // [rsp+168h] [rbp-8E8h]
  _BYTE v72[512]; // [rsp+170h] [rbp-8E0h] BYREF
  __int64 *v73; // [rsp+370h] [rbp-6E0h] BYREF
  __int64 v74; // [rsp+378h] [rbp-6D8h]
  _BYTE v75[512]; // [rsp+380h] [rbp-6D0h] BYREF
  _BYTE *v76; // [rsp+580h] [rbp-4D0h] BYREF
  __int64 v77; // [rsp+588h] [rbp-4C8h]
  _BYTE v78[512]; // [rsp+590h] [rbp-4C0h] BYREF
  _QWORD v79[6]; // [rsp+790h] [rbp-2C0h] BYREF
  __int64 v80; // [rsp+7C0h] [rbp-290h]
  __int64 v81; // [rsp+7C8h] [rbp-288h]
  __int64 v82; // [rsp+7D0h] [rbp-280h]
  __int64 v83; // [rsp+7D8h] [rbp-278h]
  char *v84; // [rsp+7E0h] [rbp-270h]
  __int64 v85; // [rsp+7E8h] [rbp-268h]
  char v86; // [rsp+7F0h] [rbp-260h] BYREF
  __int64 v87; // [rsp+810h] [rbp-240h]
  __int64 *v88; // [rsp+818h] [rbp-238h]
  __int64 v89; // [rsp+820h] [rbp-230h]
  unsigned int v90; // [rsp+828h] [rbp-228h]
  char *v91; // [rsp+830h] [rbp-220h]
  __int64 v92; // [rsp+838h] [rbp-218h]
  char v93; // [rsp+840h] [rbp-210h] BYREF
  __int64 v94; // [rsp+940h] [rbp-110h]
  __int64 *v95; // [rsp+948h] [rbp-108h]
  __int64 *v96; // [rsp+950h] [rbp-100h]
  __int64 v97; // [rsp+958h] [rbp-F8h]
  int v98; // [rsp+960h] [rbp-F0h]
  _BYTE v99[64]; // [rsp+968h] [rbp-E8h] BYREF
  __int64 v100; // [rsp+9A8h] [rbp-A8h]
  _BYTE *v101; // [rsp+9B0h] [rbp-A0h]
  _BYTE *v102; // [rsp+9B8h] [rbp-98h]
  __int64 v103; // [rsp+9C0h] [rbp-90h]
  int v104; // [rsp+9C8h] [rbp-88h]
  _BYTE v105[64]; // [rsp+9D0h] [rbp-80h] BYREF
  __int64 v106; // [rsp+A10h] [rbp-40h]
  __int64 v107; // [rsp+A18h] [rbp-38h]

  v80 = 0;
  v79[1] = 8;
  v79[0] = sub_22077B0(64);
  v12 = v79[0] + 24LL;
  v13 = sub_22077B0(512);
  v79[5] = v79[0] + 24LL;
  v85 = 0x400000000LL;
  v79[4] = v13 + 512;
  v82 = v13 + 512;
  v84 = &v86;
  v91 = &v93;
  v95 = (__int64 *)v99;
  v96 = (__int64 *)v99;
  *(_QWORD *)(v79[0] + 24LL) = v13;
  v79[3] = v13;
  v83 = v12;
  v81 = v13;
  v79[2] = v13;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v92 = 0x2000000000LL;
  v94 = 0;
  v97 = 8;
  v98 = 0;
  v100 = 0;
  v101 = v105;
  v102 = v105;
  v103 = 8;
  v104 = 0;
  v106 = a2;
  v107 = a3;
  if ( v13 )
  {
    *(_QWORD *)v13 = 0;
    *(_QWORD *)(v13 + 8) = 0;
    *(_QWORD *)(v13 + 16) = 0;
    *(_DWORD *)(v13 + 24) = 0;
  }
  v14 = a1;
  v80 = v13 + 32;
  v76 = v78;
  v77 = 0;
  v19 = sub_1AC8A90(v79, a1, v65, &v76);
  if ( v76 != v78 )
    _libc_free((unsigned __int64)v76);
  if ( (_BYTE)v19 )
  {
    v76 = v78;
    v70 = (__int64 *)v72;
    v71 = 0x2000000000LL;
    v74 = 0x2000000000LL;
    v77 = 0x2000000000LL;
    v23 = v89;
    v73 = (__int64 *)v75;
    if ( (unsigned int)v89 > 0x20 )
    {
      sub_16CD150((__int64)&v76, v78, (unsigned int)v89, 16, v17, (int)v18);
      v23 = v89;
    }
    if ( v23 )
    {
      v48 = v88;
      v18 = &v88[2 * v90];
      if ( v88 != v18 )
      {
        while ( 1 )
        {
          v49 = *v48;
          if ( *v48 != -8 && v49 != -16 )
            break;
          v48 += 2;
          if ( v18 == v48 )
            goto LABEL_10;
        }
        if ( v18 != v48 )
        {
          v50 = v48;
          v17 = v48[1];
          if ( *(_BYTE *)(v49 + 16) != 3 )
            goto LABEL_65;
LABEL_55:
          v51 = (unsigned int)v71;
          if ( (unsigned int)v71 >= HIDWORD(v71) )
          {
            v57 = v18;
            v60 = v17;
            sub_16CD150((__int64)&v70, v72, 0, 16, v17, (int)v18);
            v51 = (unsigned int)v71;
            v18 = v57;
            v17 = v60;
          }
          v52 = &v70[2 * v51];
          *v52 = v49;
          v52[1] = v17;
          LODWORD(v71) = v71 + 1;
          while ( 1 )
          {
            v50 += 2;
            if ( v50 == v18 )
              break;
            while ( 1 )
            {
              v49 = *v50;
              if ( *v50 != -8 && v49 != -16 )
                break;
              v50 += 2;
              if ( v18 == v50 )
                goto LABEL_10;
            }
            if ( v18 == v50 )
              break;
            v17 = v50[1];
            if ( *(_BYTE *)(v49 + 16) == 3 )
              goto LABEL_55;
LABEL_65:
            if ( (*(_DWORD *)(v49 + 20) & 0xFFFFFFFu) <= 3 )
            {
              v55 = (unsigned int)v77;
              if ( (unsigned int)v77 >= HIDWORD(v77) )
              {
                v59 = v18;
                v62 = v17;
                sub_16CD150((__int64)&v76, v78, 0, 16, v17, (int)v18);
                v55 = (unsigned int)v77;
                v18 = v59;
                v17 = v62;
              }
              v56 = (__int64 *)&v76[16 * v55];
              *v56 = v49;
              v56[1] = v17;
              LODWORD(v77) = v77 + 1;
            }
            else
            {
              v53 = (unsigned int)v74;
              if ( (unsigned int)v74 >= HIDWORD(v74) )
              {
                v58 = v18;
                v61 = v17;
                sub_16CD150((__int64)&v73, v75, 0, 16, v17, (int)v18);
                v53 = (unsigned int)v74;
                v18 = v58;
                v17 = v61;
              }
              v54 = &v73[2 * v53];
              *v54 = v49;
              v54[1] = v17;
              LODWORD(v74) = v74 + 1;
            }
          }
        }
      }
    }
LABEL_10:
    v24 = v73;
    v25 = &v73[2 * (unsigned int)v74];
    if ( v73 != v25 )
    {
      v64 = v19;
      do
      {
        v26 = *(_OWORD *)v24;
        if ( *(_BYTE *)(*v24 + 16) != 3 )
        {
          v27 = *(_QWORD *)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
          *(_QWORD *)&v26 = *(_QWORD *)(v27 - 24);
          v28 = sub_185BE50(v26, *v24, 2u);
          *(_QWORD *)&v26 = v27;
          *((_QWORD *)&v26 + 1) = v28;
        }
        sub_15E5440(v26, *((__int64 *)&v26 + 1));
        v24 += 2;
      }
      while ( v25 != v24 );
      v19 = v64;
    }
    v29 = v70;
    v30 = &v70[2 * (unsigned int)v71];
    if ( v70 != v30 )
    {
      do
      {
        v31 = v29[1];
        v32 = *v29;
        v29 += 2;
        sub_15E5440(v32, v31);
      }
      while ( v30 != v29 );
    }
    if ( (_DWORD)v77 )
    {
      v33 = (unsigned int)v77;
      v68[0] = (unsigned __int64)v69;
      v68[1] = 0x2000000000LL;
      if ( (unsigned int)v77 > 0x20 )
      {
        sub_16CD150((__int64)v68, v69, (unsigned int)v77, 8, v17, (int)v18);
        v33 = (unsigned int)v77;
      }
      v34 = (__int64)v76;
      v35 = 16 * v33;
      v66 = 0;
      v67[0] = &v66;
      v36 = &v76[v35];
      v67[1] = v68;
      if ( v76 == &v76[v35] )
      {
        v45 = 0;
      }
      else
      {
        v37 = 0;
        v38 = v76;
        v63 = v19;
        v39 = v36;
        while ( 1 )
        {
          v41 = *v38;
          v42 = v38[1];
          v43 = *(_QWORD *)(*v38 - 24LL * (*(_DWORD *)(*v38 + 20LL) & 0xFFFFFFF));
          sub_185B450((__int64)v67, v43, v43 != v37, v34);
          v44 = *(_QWORD *)(v41 + 24 * (2LL - (*(_DWORD *)(v41 + 20) & 0xFFFFFFF)));
          v40 = *(_DWORD *)(v44 + 32) > 0x40u ? **(_QWORD **)(v44 + 24) : *(_QWORD *)(v44 + 24);
          v38 += 2;
          *(_QWORD *)(v68[0] + 8 * v40) = v42;
          if ( v39 == v38 )
            break;
          v37 = v66;
        }
        v19 = v63;
        v45 = v66;
      }
      sub_185B450((__int64)v67, v45, 1, v34);
      if ( (_BYTE *)v68[0] != v69 )
        _libc_free(v68[0]);
    }
    if ( v76 != v78 )
      _libc_free((unsigned __int64)v76);
    if ( v73 != (__int64 *)v75 )
      _libc_free((unsigned __int64)v73);
    if ( v70 != (__int64 *)v72 )
      _libc_free((unsigned __int64)v70);
    v46 = v96;
    if ( v96 == v95 )
    {
      v15 = HIDWORD(v97);
      v14 = (__int64)&v96[HIDWORD(v97)];
    }
    else
    {
      v15 = (unsigned int)v97;
      v14 = (__int64)&v96[(unsigned int)v97];
    }
    if ( v96 != (__int64 *)v14 )
    {
      while ( 1 )
      {
        v16 = *v46;
        v15 = (__int64)v46;
        if ( (unsigned __int64)*v46 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( (__int64 *)v14 == ++v46 )
          goto LABEL_6;
      }
      while ( v14 != v15 )
      {
        v47 = (__int64 *)(v15 + 8);
        *(_BYTE *)(v16 + 80) |= 1u;
        if ( v15 + 8 == v14 )
          break;
        while ( 1 )
        {
          v16 = *v47;
          v15 = (__int64)v47;
          if ( (unsigned __int64)*v47 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( (__int64 *)v14 == ++v47 )
            goto LABEL_6;
        }
      }
    }
  }
LABEL_6:
  sub_185D1A0((__int64)v79, v14, v15, v16, a4, a5, a6, a7, v20, v21, a10, a11);
  return v19;
}
