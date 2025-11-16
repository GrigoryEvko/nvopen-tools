// Function: sub_F75950
// Address: 0xf75950
//
__int64 __fastcall sub_F75950(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 **a4,
        __int64 (__fastcall *a5)(__int64, _BYTE **, _QWORD),
        __int64 a6,
        int a7)
{
  __int64 v7; // r14
  __int64 v9; // r12
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  __m128i v13; // xmm3
  __int64 *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r13
  __int64 v18; // r15
  unsigned int v19; // eax
  __int64 v20; // rbx
  _QWORD *v21; // rax
  __int64 v22; // rax
  __int64 v23; // r13
  int v24; // r8d
  unsigned __int64 v25; // rax
  unsigned int i; // eax
  _QWORD *v27; // rcx
  unsigned int v28; // eax
  __int64 v29; // r10
  unsigned int v30; // esi
  __int64 v31; // r9
  unsigned int v32; // edx
  __int64 v33; // r8
  int v34; // r11d
  unsigned __int64 v35; // r15
  __int64 *v36; // rax
  unsigned int k; // edi
  __int64 *v38; // rdx
  __int64 v39; // rcx
  _BYTE *v40; // r12
  _BYTE *v41; // r15
  __int64 v42; // rdx
  unsigned int v43; // esi
  __int64 v44; // rdi
  __int64 v45; // rsi
  _QWORD *v47; // rax
  _QWORD *v48; // r10
  _QWORD **v49; // rdx
  int v50; // ecx
  __int64 *v51; // rax
  __int64 v52; // rax
  __int64 v53; // r13
  _BYTE *v54; // rbx
  _BYTE *v55; // r15
  __int64 v56; // rdx
  unsigned int v57; // esi
  __int64 *v58; // rdi
  unsigned int j; // edx
  int v60; // edx
  __int64 v61; // r15
  __int64 v62; // rdx
  _QWORD *v63; // rax
  __int64 v64; // rbx
  _BYTE *v65; // r15
  _BYTE *v66; // r13
  __int64 v67; // rdx
  unsigned int v68; // esi
  _BYTE *v69; // r13
  _BYTE *v70; // rbx
  __int64 v71; // rdx
  unsigned int v72; // esi
  unsigned int v73; // edi
  int v74; // ecx
  __int64 *v75; // rsi
  unsigned int v76; // r15d
  int m; // edi
  unsigned int v78; // r15d
  __int64 v79; // [rsp+18h] [rbp-238h]
  __int64 v80; // [rsp+28h] [rbp-228h]
  __int64 *v83; // [rsp+40h] [rbp-210h]
  __int64 v84; // [rsp+48h] [rbp-208h]
  __int64 v86; // [rsp+60h] [rbp-1F0h]
  __int64 v87; // [rsp+68h] [rbp-1E8h]
  __int64 v88; // [rsp+68h] [rbp-1E8h]
  __int64 v89; // [rsp+68h] [rbp-1E8h]
  __int64 v90; // [rsp+68h] [rbp-1E8h]
  __int64 v91; // [rsp+68h] [rbp-1E8h]
  __int64 v92; // [rsp+68h] [rbp-1E8h]
  __int64 v93; // [rsp+68h] [rbp-1E8h]
  __int64 v94; // [rsp+68h] [rbp-1E8h]
  __int64 v95; // [rsp+68h] [rbp-1E8h]
  __int64 v96; // [rsp+78h] [rbp-1D8h]
  __int64 v97; // [rsp+80h] [rbp-1D0h] BYREF
  __int64 v98; // [rsp+88h] [rbp-1C8h]
  __int64 v99; // [rsp+90h] [rbp-1C0h]
  unsigned int v100; // [rsp+98h] [rbp-1B8h]
  _QWORD v101[4]; // [rsp+A0h] [rbp-1B0h] BYREF
  __int16 v102; // [rsp+C0h] [rbp-190h]
  _QWORD v103[2]; // [rsp+D0h] [rbp-180h] BYREF
  __int64 v104; // [rsp+E0h] [rbp-170h]
  __m128i v105; // [rsp+E8h] [rbp-168h] BYREF
  __m128i v106; // [rsp+F8h] [rbp-158h] BYREF
  __m128i v107; // [rsp+108h] [rbp-148h] BYREF
  __m128i v108; // [rsp+118h] [rbp-138h] BYREF
  __int64 v109; // [rsp+128h] [rbp-128h]
  _BYTE *v110; // [rsp+130h] [rbp-120h] BYREF
  __int64 v111; // [rsp+138h] [rbp-118h]
  _BYTE v112[32]; // [rsp+140h] [rbp-110h] BYREF
  __int64 v113; // [rsp+160h] [rbp-F0h]
  __int64 v114; // [rsp+168h] [rbp-E8h]
  __int64 v115; // [rsp+170h] [rbp-E0h]
  __int64 v116; // [rsp+178h] [rbp-D8h]
  void **v117; // [rsp+180h] [rbp-D0h]
  void **v118; // [rsp+188h] [rbp-C8h]
  __int64 v119; // [rsp+190h] [rbp-C0h]
  int v120; // [rsp+198h] [rbp-B8h]
  __int16 v121; // [rsp+19Ch] [rbp-B4h]
  char v122; // [rsp+19Eh] [rbp-B2h]
  __int64 v123; // [rsp+1A0h] [rbp-B0h]
  __int64 v124; // [rsp+1A8h] [rbp-A8h]
  void *v125; // [rsp+1B0h] [rbp-A0h] BYREF
  void *v126; // [rsp+1B8h] [rbp-98h]
  __int64 v127; // [rsp+1C0h] [rbp-90h]
  __m128i v128; // [rsp+1C8h] [rbp-88h]
  __m128i v129; // [rsp+1D8h] [rbp-78h]
  __m128i v130; // [rsp+1E8h] [rbp-68h]
  __m128i v131; // [rsp+1F8h] [rbp-58h]
  __int64 v132; // [rsp+208h] [rbp-48h]
  void *v133; // [rsp+210h] [rbp-40h] BYREF

  v9 = a2;
  v110 = v112;
  v117 = &v125;
  v116 = sub_BD5C60(a1);
  v118 = &v133;
  v103[0] = &unk_49E5698;
  v103[1] = &unk_49D94D0;
  LOWORD(v109) = 257;
  v111 = 0x200000000LL;
  v121 = 512;
  v104 = sub_B43CC0(a1);
  v105 = (__m128i)(unsigned __int64)v104;
  LOWORD(v115) = 0;
  v106 = 0u;
  v107 = 0u;
  v108 = 0u;
  v119 = 0;
  v120 = 0;
  v122 = 7;
  v123 = 0;
  v124 = 0;
  v113 = 0;
  v114 = 0;
  v125 = &unk_49E5698;
  v10 = _mm_loadu_si128(&v105);
  v127 = v104;
  v11 = _mm_loadu_si128(&v106);
  v12 = _mm_loadu_si128(&v107);
  v126 = &unk_49D94D0;
  v132 = v109;
  v13 = _mm_loadu_si128(&v108);
  v128 = v10;
  v129 = v11;
  v133 = &unk_49DA0B0;
  v130 = v12;
  v131 = v13;
  nullsub_63();
  nullsub_63();
  sub_D5F1F0((__int64)&v110, a1);
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v14 = *a4;
  v100 = 0;
  v83 = v14;
  v84 = a2 + 24 * a3;
  if ( v84 != a2 )
  {
    v86 = 0;
    v80 = a1 + 24;
    while ( 1 )
    {
      v15 = sub_D95540(*(_QWORD *)(v9 + 8));
      v16 = (unsigned int)(*(_DWORD *)(v9 + 16) * a7);
      v17 = v15;
      v102 = 257;
      v18 = sub_AD64C0(v15, v16, 0);
      v19 = sub_BCB060(v17);
      v87 = a5(a6, &v110, v19);
      v20 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, _QWORD, _QWORD))*v117 + 4))(
              v117,
              17,
              v87,
              v18,
              0,
              0);
      if ( !v20 )
      {
        v105.m128i_i16[4] = 257;
        v20 = sub_B504D0(17, v87, v18, (__int64)v103, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v118 + 2))(v118, v20, v101, v114, v115);
        if ( v110 != &v110[16 * (unsigned int)v111] )
        {
          v88 = v9;
          v40 = v110;
          v41 = &v110[16 * (unsigned int)v111];
          do
          {
            v42 = *((_QWORD *)v40 + 1);
            v43 = *(_DWORD *)v40;
            v40 += 16;
            sub_B99FD0(v20, v43, v42);
          }
          while ( v41 != v40 );
          v9 = v88;
        }
      }
      LOWORD(v7) = 0;
      v21 = sub_DCC810(v83, *(_QWORD *)(v9 + 8), *(_QWORD *)v9, 0, 0);
      v22 = sub_F8DB90(a4, v21, v17, v80, v7);
      v23 = v22;
      if ( v100 )
      {
        v24 = 1;
        v25 = 0xBF58476D1CE4E5B9LL
            * (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4)
             | ((unsigned __int64)(((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4)) << 32));
        for ( i = (v100 - 1) & ((v25 >> 31) ^ v25); ; i = (v100 - 1) & v28 )
        {
          v27 = (_QWORD *)(v98 + 24LL * i);
          if ( v23 == *v27 && v27[1] == v20 )
            break;
          if ( *v27 == -4096 && v27[1] == -4096 )
            goto LABEL_10;
          v28 = v24 + i;
          ++v24;
        }
        if ( v27[2] )
          goto LABEL_26;
      }
LABEL_10:
      v101[0] = "diff.check";
      v102 = 259;
      v29 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v117 + 7))(v117, 36, v23, v20);
      if ( v29 )
        goto LABEL_11;
      v105.m128i_i16[4] = 257;
      v47 = sub_BD2C40(72, unk_3F10FD0);
      v48 = v47;
      if ( v47 )
      {
        v49 = *(_QWORD ***)(v23 + 8);
        v89 = (__int64)v47;
        v50 = *((unsigned __int8 *)v49 + 8);
        if ( (unsigned int)(v50 - 17) > 1 )
        {
          v52 = sub_BCB2A0(*v49);
        }
        else
        {
          BYTE4(v96) = (_BYTE)v50 == 18;
          LODWORD(v96) = *((_DWORD *)v49 + 8);
          v51 = (__int64 *)sub_BCB2A0(*v49);
          v52 = sub_BCE1B0(v51, v96);
        }
        sub_B523C0(v89, v52, 53, 36, v23, v20, (__int64)v103, 0, 0, 0);
        v48 = (_QWORD *)v89;
      }
      v90 = (__int64)v48;
      (*((void (__fastcall **)(void **, _QWORD *, _QWORD *, __int64, __int64))*v118 + 2))(v118, v48, v101, v114, v115);
      v29 = v90;
      if ( v110 == &v110[16 * (unsigned int)v111] )
      {
LABEL_11:
        v30 = v100;
        if ( !v100 )
          goto LABEL_39;
      }
      else
      {
        v91 = v23;
        v53 = v29;
        v79 = v20;
        v54 = v110;
        v55 = &v110[16 * (unsigned int)v111];
        do
        {
          v56 = *((_QWORD *)v54 + 1);
          v57 = *(_DWORD *)v54;
          v54 += 16;
          sub_B99FD0(v53, v57, v56);
        }
        while ( v55 != v54 );
        v30 = v100;
        v29 = v53;
        v20 = v79;
        v23 = v91;
        if ( !v100 )
        {
LABEL_39:
          ++v97;
LABEL_40:
          v92 = v29;
          sub_F75680((__int64)&v97, 2 * v30);
          if ( v100 )
          {
            v29 = v92;
            v33 = 1;
            v58 = 0;
            for ( j = (v100 - 1)
                    & (((0xBF58476D1CE4E5B9LL
                       * (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4)
                        | ((unsigned __int64)(((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)) << 32))) >> 31)
                     ^ (484763065 * (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4)))); ; j = (v100 - 1) & v60 )
            {
              v36 = (__int64 *)(v98 + 24LL * j);
              v31 = *v36;
              if ( v23 == *v36 && v20 == v36[1] )
                break;
              if ( v31 == -4096 )
              {
                if ( v36[1] == -4096 )
                {
                  v74 = v99 + 1;
                  if ( v58 )
                    v36 = v58;
                  goto LABEL_70;
                }
              }
              else if ( v31 == -8192 && v36[1] == -8192 && !v58 )
              {
                v58 = (__int64 *)(v98 + 24LL * j);
              }
              v60 = v33 + j;
              v33 = (unsigned int)(v33 + 1);
            }
            goto LABEL_76;
          }
LABEL_96:
          LODWORD(v99) = v99 + 1;
          BUG();
        }
      }
      v31 = v98;
      v32 = (unsigned int)v20 >> 9;
      v33 = v30 - 1;
      v34 = 1;
      v35 = ((0xBF58476D1CE4E5B9LL
            * (v32 ^ ((unsigned int)v20 >> 4)
             | ((unsigned __int64)(((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)) << 32))) >> 31)
          ^ (0xBF58476D1CE4E5B9LL
           * (v32 ^ ((unsigned int)v20 >> 4)
            | ((unsigned __int64)(((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)) << 32)));
      v36 = 0;
      for ( k = (((0xBF58476D1CE4E5B9LL
                 * (v32 ^ ((unsigned int)v20 >> 4)
                  | ((unsigned __int64)(((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)) << 32))) >> 31)
               ^ (484763065 * (v32 ^ ((unsigned int)v20 >> 4))))
              & (v30 - 1); ; k = v33 & v73 )
      {
        v38 = (__int64 *)(v98 + 24LL * k);
        v39 = *v38;
        if ( v23 == *v38 && v20 == v38[1] )
          goto LABEL_50;
        if ( v39 == -4096 )
          break;
        if ( v39 == -8192 && v38[1] == -8192 && !v36 )
          v36 = (__int64 *)(v98 + 24LL * k);
LABEL_65:
        v73 = v34 + k;
        ++v34;
      }
      if ( v38[1] != -4096 )
        goto LABEL_65;
      if ( !v36 )
        v36 = (__int64 *)(v98 + 24LL * k);
      ++v97;
      v74 = v99 + 1;
      if ( 4 * ((int)v99 + 1) >= 3 * v30 )
        goto LABEL_40;
      if ( v30 - HIDWORD(v99) - v74 > v30 >> 3 )
        goto LABEL_70;
      v95 = v29;
      sub_F75680((__int64)&v97, v30);
      if ( !v100 )
        goto LABEL_96;
      v29 = v95;
      v75 = 0;
      v76 = (v100 - 1) & v35;
      for ( m = 1; ; ++m )
      {
        v36 = (__int64 *)(v98 + 24LL * v76);
        v33 = *v36;
        if ( v23 == *v36 && v20 == v36[1] )
          break;
        if ( v33 == -4096 )
        {
          if ( v36[1] == -4096 )
          {
            v74 = v99 + 1;
            if ( v75 )
              v36 = v75;
            goto LABEL_70;
          }
        }
        else if ( v33 == -8192 && v36[1] == -8192 && !v75 )
        {
          v75 = (__int64 *)(v98 + 24LL * v76);
        }
        v78 = m + v76;
        v76 = (v100 - 1) & v78;
      }
LABEL_76:
      v74 = v99 + 1;
LABEL_70:
      LODWORD(v99) = v74;
      if ( *v36 != -4096 || v36[1] != -4096 )
        --HIDWORD(v99);
      *v36 = v23;
      v36[1] = v20;
      v36[2] = v29;
LABEL_50:
      if ( *(_BYTE *)(v9 + 20) )
      {
        v94 = v29;
        v101[0] = sub_BD5D20(v29);
        v101[2] = ".fr";
        v105.m128i_i16[4] = 257;
        v102 = 773;
        v101[1] = v62;
        v63 = sub_BD2C40(72, unk_3F10A14);
        v64 = (__int64)v63;
        if ( v63 )
          sub_B549F0((__int64)v63, v94, (__int64)v103, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v118 + 2))(v118, v64, v101, v114, v115);
        v65 = v110;
        v66 = &v110[16 * (unsigned int)v111];
        if ( v110 != v66 )
        {
          do
          {
            v67 = *((_QWORD *)v65 + 1);
            v68 = *(_DWORD *)v65;
            v65 += 16;
            sub_B99FD0(v64, v68, v67);
          }
          while ( v66 != v65 );
        }
        v29 = v64;
      }
      if ( v86 )
      {
        v101[0] = "conflict.rdx";
        v102 = 259;
        v93 = v29;
        v61 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, __int64, __int64))*v117 + 2))(
                v117,
                29,
                v86,
                v29,
                v33,
                v31);
        if ( !v61 )
        {
          v105.m128i_i16[4] = 257;
          v61 = sub_B504D0(29, v86, v93, (__int64)v103, 0, 0);
          (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v118 + 2))(
            v118,
            v61,
            v101,
            v114,
            v115);
          v69 = v110;
          v70 = &v110[16 * (unsigned int)v111];
          if ( v110 != v70 )
          {
            do
            {
              v71 = *((_QWORD *)v69 + 1);
              v72 = *(_DWORD *)v69;
              v69 += 16;
              sub_B99FD0(v61, v72, v71);
            }
            while ( v70 != v69 );
          }
        }
        v86 = v61;
      }
      else
      {
        v86 = v29;
      }
LABEL_26:
      v9 += 24;
      if ( v84 == v9 )
      {
        v44 = v98;
        v45 = 24LL * v100;
        goto LABEL_28;
      }
    }
  }
  v86 = 0;
  v44 = 0;
  v45 = 0;
LABEL_28:
  sub_C7D6A0(v44, v45, 8);
  nullsub_61();
  v125 = &unk_49E5698;
  v126 = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  if ( v110 != v112 )
    _libc_free(v110, v45);
  return v86;
}
