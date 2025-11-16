// Function: sub_2A28FB0
// Address: 0x2a28fb0
//
__int64 __fastcall sub_2A28FB0(__int64 *a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // r12
  __int64 v4; // r15
  __int64 v5; // r13
  unsigned __int64 v6; // rax
  __int64 *v7; // rsi
  unsigned __int64 v8; // rax
  int v9; // edx
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // r13
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  int v15; // edx
  __int64 v16; // rax
  bool v17; // cf
  __int64 v18; // rdx
  _QWORD *v19; // r14
  __m128i v20; // xmm0
  __m128i v21; // xmm1
  __m128i v22; // xmm2
  __m128i v23; // xmm3
  const char *v24; // rax
  __int64 v25; // rdx
  const char *v26; // rax
  __int64 v27; // r8
  __int64 v28; // rcx
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // r9
  __int64 v34; // rdx
  _QWORD *v35; // rax
  __int64 v36; // rsi
  unsigned __int64 v37; // rax
  int v38; // edx
  __int64 v39; // r15
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // r15
  _QWORD *v43; // rax
  __int64 v44; // r9
  __int64 v45; // r14
  unsigned __int64 v46; // r15
  _BYTE *v47; // r13
  __int64 v48; // rdx
  unsigned int v49; // esi
  __int64 v50; // r13
  __int64 v51; // r14
  __int64 v52; // rax
  __int64 v53; // r8
  __int64 v54; // r9
  unsigned int v55; // ecx
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rdx
  unsigned int v59; // eax
  __int64 v60; // r12
  __int64 v61; // r14
  _QWORD *v62; // rdi
  __int64 v63; // rdx
  _QWORD *v64; // rcx
  __int64 v65; // rsi
  _QWORD *v66; // rax
  _QWORD *v67; // rsi
  __int64 v68; // rax
  char **v69; // rdi
  __int64 v70; // rax
  __int64 v71; // rdx
  char *v72; // rdx
  char **v73; // r12
  char **v74; // r15
  char *v75; // r14
  unsigned __int64 v77; // rax
  int v78; // edx
  __int64 v79; // rsi
  __int64 v80; // rax
  __int64 v81; // rax
  unsigned __int64 v82; // rbx
  _BYTE *v83; // r14
  __int64 v84; // r13
  __int64 v85; // rdx
  unsigned int v86; // esi
  _QWORD *v87; // rax
  __int64 v88; // [rsp-20h] [rbp-AE0h]
  __int64 v89; // [rsp+10h] [rbp-AB0h]
  _QWORD *v91; // [rsp+40h] [rbp-A80h]
  __int64 v92; // [rsp+40h] [rbp-A80h]
  __int64 *v93; // [rsp+40h] [rbp-A80h]
  const char *v94; // [rsp+60h] [rbp-A60h] BYREF
  __int64 v95; // [rsp+68h] [rbp-A58h]
  _BYTE v96[64]; // [rsp+70h] [rbp-A50h] BYREF
  _BYTE *v97; // [rsp+B0h] [rbp-A10h] BYREF
  __int64 v98; // [rsp+B8h] [rbp-A08h]
  _BYTE v99[32]; // [rsp+C0h] [rbp-A00h] BYREF
  __int64 v100; // [rsp+E0h] [rbp-9E0h]
  __int64 v101; // [rsp+E8h] [rbp-9D8h]
  __int64 v102; // [rsp+F0h] [rbp-9D0h]
  __int64 v103; // [rsp+F8h] [rbp-9C8h]
  void **v104; // [rsp+100h] [rbp-9C0h]
  void **v105; // [rsp+108h] [rbp-9B8h]
  __int64 v106; // [rsp+110h] [rbp-9B0h]
  int v107; // [rsp+118h] [rbp-9A8h]
  __int16 v108; // [rsp+11Ch] [rbp-9A4h]
  char v109; // [rsp+11Eh] [rbp-9A2h]
  __int64 v110; // [rsp+120h] [rbp-9A0h]
  __int64 v111; // [rsp+128h] [rbp-998h]
  void *v112; // [rsp+130h] [rbp-990h] BYREF
  void *v113; // [rsp+138h] [rbp-988h]
  char *v114; // [rsp+140h] [rbp-980h]
  __m128i v115; // [rsp+148h] [rbp-978h]
  __m128i v116; // [rsp+158h] [rbp-968h]
  __m128i v117; // [rsp+168h] [rbp-958h]
  __m128i v118; // [rsp+178h] [rbp-948h]
  __int64 v119; // [rsp+188h] [rbp-938h]
  void *v120; // [rsp+190h] [rbp-930h] BYREF
  char **v121; // [rsp+1A0h] [rbp-920h] BYREF
  __int64 v122; // [rsp+1A8h] [rbp-918h]
  char *v123; // [rsp+1B0h] [rbp-910h] BYREF
  __m128i v124; // [rsp+1B8h] [rbp-908h] BYREF
  __m128i v125; // [rsp+1C8h] [rbp-8F8h] BYREF
  __m128i v126; // [rsp+1D8h] [rbp-8E8h] BYREF
  __m128i v127; // [rsp+1E8h] [rbp-8D8h] BYREF
  __int64 v128; // [rsp+1F8h] [rbp-8C8h]
  __int64 v129[110]; // [rsp+3B0h] [rbp-710h] BYREF
  _BYTE v130[928]; // [rsp+720h] [rbp-3A0h] BYREF

  v2 = a1;
  v3 = sub_D4B130(*a1);
  v4 = v3 + 48;
  v5 = *(_QWORD *)(a1[35] + 8);
  v6 = sub_AA4E30(**(_QWORD **)(*a1 + 32));
  sub_27C1C30((__int64)v129, *(__int64 **)(v5 + 288), v6, (__int64)"induction", 1);
  v7 = (__int64 *)*a1;
  v8 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 + 48 == v8 )
  {
    v10 = 0;
  }
  else
  {
    if ( !v8 )
      BUG();
    v9 = *(unsigned __int8 *)(v8 - 24);
    v10 = 0;
    v11 = v8 - 24;
    if ( (unsigned int)(v9 - 30) < 0xB )
      v10 = v11;
  }
  v12 = sub_F73BC0(v10, v7, (__int64)(v2 + 12), v129, 0);
  v13 = sub_AA4E30(v3);
  sub_27C1C30((__int64)v130, (__int64 *)v2[38], v13, (__int64)"scev.check", 1);
  v14 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v14 == v4 )
  {
    v18 = 0;
  }
  else
  {
    if ( !v14 )
      BUG();
    v15 = *(unsigned __int8 *)(v14 - 24);
    v16 = v14 - 24;
    v17 = (unsigned int)(v15 - 30) < 0xB;
    v18 = 0;
    if ( v17 )
      v18 = v16;
  }
  v19 = sub_F8C220((__int64)v130, v2[22], v18);
  LOWORD(v128) = 257;
  v123 = (char *)sub_AA4E30(v3);
  v121 = (char **)&unk_49E5698;
  v122 = (__int64)&unk_49D94D0;
  v124 = (__m128i)(unsigned __int64)v123;
  v125 = 0u;
  v126 = 0u;
  v127 = 0u;
  v103 = sub_AA48A0(v3);
  v20 = _mm_loadu_si128(&v124);
  v21 = _mm_loadu_si128(&v125);
  v104 = &v112;
  v22 = _mm_loadu_si128(&v126);
  v23 = _mm_loadu_si128(&v127);
  v105 = &v120;
  v114 = v123;
  v97 = v99;
  v98 = 0x200000000LL;
  v108 = 512;
  LOWORD(v102) = 0;
  v112 = &unk_49E5698;
  v113 = &unk_49D94D0;
  v115 = v20;
  v116 = v21;
  v117 = v22;
  v118 = v23;
  v106 = 0;
  v107 = 0;
  v109 = 7;
  v110 = 0;
  v111 = 0;
  v100 = 0;
  v101 = 0;
  v119 = v128;
  v121 = (char **)&unk_49E5698;
  v122 = (__int64)&unk_49D94D0;
  v120 = &unk_49DA0B0;
  nullsub_63();
  nullsub_63();
  if ( v12 && v19 )
  {
    v77 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v4 == v77 )
    {
      v79 = 0;
    }
    else
    {
      if ( !v77 )
        BUG();
      v78 = *(unsigned __int8 *)(v77 - 24);
      v79 = 0;
      v80 = v77 - 24;
      if ( (unsigned int)(v78 - 30) < 0xB )
        v79 = v80;
    }
    sub_D5F1F0((__int64)&v97, v79);
    v96[17] = 1;
    v94 = "lver.safe";
    v96[16] = 3;
    v81 = (*((__int64 (__fastcall **)(void **, __int64, __int64, _QWORD *))*v104 + 2))(v104, 29, v12, v19);
    if ( !v81 )
    {
      v124.m128i_i16[4] = 257;
      v92 = sub_B504D0(29, v12, (__int64)v19, (__int64)&v121, 0, 0);
      (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v105 + 2))(
        v105,
        v92,
        &v94,
        v101,
        v102);
      v81 = v92;
      if ( v97 != &v97[16 * (unsigned int)v98] )
      {
        v93 = v2;
        v82 = (unsigned __int64)v97;
        v83 = &v97[16 * (unsigned int)v98];
        v84 = v81;
        do
        {
          v85 = *(_QWORD *)(v82 + 8);
          v86 = *(_DWORD *)v82;
          v82 += 16LL;
          sub_B99FD0(v84, v86, v85);
        }
        while ( v83 != (_BYTE *)v82 );
        v2 = v93;
        v81 = v84;
      }
    }
    v12 = v81;
  }
  else if ( !v12 )
  {
    v12 = (__int64)v19;
  }
  v24 = sub_BD5D20(**(_QWORD **)(*v2 + 32));
  v124.m128i_i16[4] = 773;
  v121 = (char **)v24;
  v122 = v25;
  v123 = ".lver.check";
  sub_BD6B50((unsigned __int8 *)v3, (const char **)&v121);
  v26 = sub_BD5D20(**(_QWORD **)(*v2 + 32));
  v27 = v2[36];
  v28 = v2[37];
  v121 = (char **)v26;
  v124.m128i_i16[4] = 773;
  v122 = v29;
  v123 = ".ph";
  v30 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 == v30 )
  {
    v31 = 0;
  }
  else
  {
    if ( !v30 )
      BUG();
    v31 = v30 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v30 - 24) - 30 >= 0xB )
      v31 = 0;
  }
  v32 = sub_F36960(v3, (__int64 *)(v31 + 24), 0, v28, v27, 0, (void **)&v121, 0);
  v94 = v96;
  v88 = v2[37];
  v33 = v2[36];
  v34 = *v2;
  v95 = 0x800000000LL;
  v121 = (char **)".lver.orig";
  v124.m128i_i16[4] = 259;
  v35 = sub_F4CEE0(v32, v3, v34, (__int64)(v2 + 2), (__int64 *)&v121, v33, v88, (__int64)&v94);
  v36 = (unsigned int)v95;
  v2[1] = (__int64)v35;
  sub_F45F60((__int64)v94, v36, (__int64)(v2 + 2));
  v37 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 == v37 )
  {
    v91 = 0;
    sub_D5F1F0((__int64)&v97, 0);
  }
  else
  {
    if ( !v37 )
      BUG();
    v38 = *(unsigned __int8 *)(v37 - 24);
    v39 = 0;
    v40 = v37 - 24;
    if ( (unsigned int)(v38 - 30) < 0xB )
      v39 = v40;
    v91 = (_QWORD *)v39;
    sub_D5F1F0((__int64)&v97, v39);
  }
  v89 = sub_D4B130(*v2);
  v41 = sub_D4B130(v2[1]);
  v124.m128i_i16[4] = 257;
  v42 = v41;
  v43 = sub_BD2C40(72, 3u);
  v45 = (__int64)v43;
  if ( v43 )
    sub_B4C9A0((__int64)v43, v42, v89, v12, 3u, v44, 0, 0);
  (*((void (__fastcall **)(void **, __int64, char ***, __int64, __int64))*v105 + 2))(v105, v45, &v121, v101, v102);
  v46 = (unsigned __int64)v97;
  v47 = &v97[16 * (unsigned int)v98];
  if ( v97 != v47 )
  {
    do
    {
      v48 = *(_QWORD *)(v46 + 8);
      v49 = *(_DWORD *)v46;
      v46 += 16LL;
      sub_B99FD0(v45, v49, v48);
    }
    while ( v47 != (_BYTE *)v46 );
  }
  v50 = 0;
  sub_B43D60(v91);
  v51 = v2[37];
  v52 = sub_D47470(*v2);
  v55 = *(_DWORD *)(v51 + 32);
  v56 = v52;
  v57 = (unsigned int)(*(_DWORD *)(v3 + 44) + 1);
  if ( (unsigned int)v57 < v55 )
    v50 = *(_QWORD *)(*(_QWORD *)(v51 + 24) + 8 * v57);
  if ( v56 )
  {
    v58 = (unsigned int)(*(_DWORD *)(v56 + 44) + 1);
    v59 = v58;
  }
  else
  {
    v58 = 0;
    v59 = 0;
  }
  if ( v55 <= v59 )
  {
    *(_BYTE *)(v51 + 112) = 0;
    BUG();
  }
  v60 = *(_QWORD *)(*(_QWORD *)(v51 + 24) + 8 * v58);
  *(_BYTE *)(v51 + 112) = 0;
  v61 = *(_QWORD *)(v60 + 8);
  if ( v50 != v61 )
  {
    v62 = *(_QWORD **)(v61 + 24);
    v63 = *(unsigned int *)(v61 + 32);
    v64 = &v62[v63];
    v65 = (8 * v63) >> 3;
    if ( (8 * v63) >> 5 )
    {
      v66 = &v62[4 * ((8 * v63) >> 5)];
      while ( v60 != *v62 )
      {
        if ( v60 == v62[1] )
        {
          v67 = ++v62 + 1;
          goto LABEL_40;
        }
        if ( v60 == v62[2] )
        {
          v62 += 2;
          v67 = v62 + 1;
          goto LABEL_40;
        }
        if ( v60 == v62[3] )
        {
          v62 += 3;
          v67 = v62 + 1;
          goto LABEL_40;
        }
        v62 += 4;
        if ( v66 == v62 )
        {
          v65 = v64 - v62;
          goto LABEL_69;
        }
      }
      goto LABEL_39;
    }
LABEL_69:
    switch ( v65 )
    {
      case 2LL:
        v87 = v62;
        break;
      case 3LL:
        v67 = v62 + 1;
        v87 = v62 + 1;
        if ( v60 == *v62 )
          goto LABEL_40;
        break;
      case 1LL:
        goto LABEL_80;
      default:
LABEL_72:
        v62 = v64;
        v67 = v64 + 1;
        goto LABEL_40;
    }
    v62 = v87 + 1;
    if ( v60 == *v87 )
    {
      v62 = v87;
      goto LABEL_39;
    }
LABEL_80:
    if ( v60 != *v62 )
      goto LABEL_72;
LABEL_39:
    v67 = v62 + 1;
LABEL_40:
    if ( v67 != v64 )
    {
      memmove(v62, v67, (char *)v64 - (char *)v67);
      LODWORD(v63) = *(_DWORD *)(v61 + 32);
    }
    *(_DWORD *)(v61 + 32) = v63 - 1;
    *(_QWORD *)(v60 + 8) = v50;
    v68 = *(unsigned int *)(v50 + 32);
    if ( v68 + 1 > (unsigned __int64)*(unsigned int *)(v50 + 36) )
    {
      sub_C8D5F0(v50 + 24, (const void *)(v50 + 40), v68 + 1, 8u, v53, v54);
      v68 = *(unsigned int *)(v50 + 32);
    }
    *(_QWORD *)(*(_QWORD *)(v50 + 24) + 8 * v68) = v60;
    ++*(_DWORD *)(v50 + 32);
    if ( *(_DWORD *)(v60 + 16) != *(_DWORD *)(*(_QWORD *)(v60 + 8) + 16LL) + 1 )
    {
      v123 = (char *)v60;
      v121 = &v123;
      v69 = &v123;
      v122 = 0x4000000001LL;
      LODWORD(v70) = 1;
      do
      {
        v71 = (unsigned int)v70;
        v70 = (unsigned int)(v70 - 1);
        v72 = v69[v71 - 1];
        LODWORD(v122) = v70;
        v73 = (char **)*((_QWORD *)v72 + 3);
        *((_DWORD *)v72 + 4) = *(_DWORD *)(*((_QWORD *)v72 + 1) + 16LL) + 1;
        v74 = &v73[*((unsigned int *)v72 + 8)];
        if ( v73 != v74 )
        {
          do
          {
            v75 = *v73;
            if ( *((_DWORD *)*v73 + 4) != *(_DWORD *)(*((_QWORD *)*v73 + 1) + 16LL) + 1 )
            {
              if ( v70 + 1 > (unsigned __int64)HIDWORD(v122) )
              {
                sub_C8D5F0((__int64)&v121, &v123, v70 + 1, 8u, v53, v54);
                v70 = (unsigned int)v122;
              }
              v121[v70] = v75;
              v70 = (unsigned int)(v122 + 1);
              LODWORD(v122) = v122 + 1;
            }
            ++v73;
          }
          while ( v74 != v73 );
          v69 = v121;
        }
      }
      while ( (_DWORD)v70 );
      if ( v69 != &v123 )
        _libc_free((unsigned __int64)v69);
    }
  }
  sub_2A28A90(v2, a2);
  sub_F6D150(v2[1], v2[37], v2[36], 0, 1);
  sub_F6D150(*v2, v2[37], v2[36], 0, 1);
  if ( v94 != v96 )
    _libc_free((unsigned __int64)v94);
  nullsub_61();
  v112 = &unk_49E5698;
  v113 = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  if ( v97 != v99 )
    _libc_free((unsigned __int64)v97);
  sub_27C20B0((__int64)v130);
  return sub_27C20B0((__int64)v129);
}
