// Function: sub_25A2620
// Address: 0x25a2620
//
__int64 __fastcall sub_25A2620(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 v4; // rsi
  __int64 result; // rax
  __int64 v6; // rbx
  _QWORD *v7; // rax
  _QWORD *v8; // r13
  __int64 *v9; // rbx
  unsigned __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned __int64 *v15; // r10
  __int64 v16; // rax
  unsigned int v17; // esi
  unsigned __int64 *v18; // r14
  int v19; // eax
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdx
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rsi
  int v25; // eax
  __int64 v26; // rcx
  __int64 *v27; // rsi
  unsigned __int64 *v28; // rdi
  __int64 v29; // rax
  unsigned __int64 v30; // rsi
  __int64 *v31; // r14
  __int64 v32; // rax
  __int64 v33; // r9
  _QWORD *v34; // rbx
  _QWORD *v35; // r14
  unsigned int v36; // esi
  unsigned __int64 *v37; // r13
  int v38; // edx
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // r9
  unsigned int v43; // eax
  unsigned __int64 *v44; // rdi
  unsigned __int64 v45; // rcx
  int v46; // r8d
  int v47; // eax
  __int64 v48; // r12
  __int64 v49; // rax
  __int64 *v50; // rax
  __int64 *v51; // rax
  __int64 v52; // rax
  __int64 v53; // r13
  unsigned __int16 v54; // ax
  unsigned __int8 v55; // bl
  __int64 v56; // rax
  unsigned __int16 v57; // ax
  unsigned __int64 v58; // rdi
  __m128i v59; // rax
  _BYTE *v60; // rax
  __int64 v61; // rdx
  char v62; // r13
  unsigned int v63; // edx
  unsigned __int64 v64; // rax
  _QWORD *v65; // rax
  __int64 *v66; // r13
  __int64 v67; // rdx
  _QWORD *v68; // rax
  __int64 v69; // r12
  __int64 v70; // rcx
  _QWORD *v71; // rax
  __int64 **v72; // rax
  unsigned __int8 *v73; // r13
  __int64 v74; // rsi
  unsigned __int16 v75; // bx
  _QWORD *v76; // rdi
  unsigned int v77; // esi
  __int64 v78; // rdi
  unsigned int v79; // eax
  unsigned __int64 *v80; // rbx
  unsigned __int64 v81; // rcx
  __int64 v82; // rax
  __int64 v83; // rax
  _QWORD *v84; // rbx
  _QWORD *v85; // r13
  __int64 v86; // rsi
  __int64 v87; // rsi
  unsigned int v88; // eax
  __int64 v89; // rax
  int v90; // r8d
  unsigned __int64 v91; // rbx
  unsigned __int16 v92; // [rsp+6h] [rbp-2DAh]
  _QWORD *v93; // [rsp+18h] [rbp-2C8h]
  _QWORD *v94; // [rsp+20h] [rbp-2C0h]
  __int64 v96; // [rsp+30h] [rbp-2B0h]
  __int64 v97; // [rsp+38h] [rbp-2A8h]
  unsigned __int8 v98; // [rsp+38h] [rbp-2A8h]
  __int64 v99; // [rsp+40h] [rbp-2A0h]
  unsigned int v100; // [rsp+40h] [rbp-2A0h]
  __int64 v101; // [rsp+40h] [rbp-2A0h]
  __int64 v102; // [rsp+48h] [rbp-298h]
  __int64 v103; // [rsp+50h] [rbp-290h]
  __int64 v104; // [rsp+50h] [rbp-290h]
  __int64 v105; // [rsp+68h] [rbp-278h]
  _QWORD *v106; // [rsp+70h] [rbp-270h]
  __int64 v107; // [rsp+70h] [rbp-270h]
  __int64 v108; // [rsp+78h] [rbp-268h]
  char v109; // [rsp+87h] [rbp-259h] BYREF
  __int64 *v110; // [rsp+88h] [rbp-258h] BYREF
  _BYTE *v111; // [rsp+90h] [rbp-250h]
  __int64 v112; // [rsp+98h] [rbp-248h]
  __int64 v113; // [rsp+A0h] [rbp-240h] BYREF
  __int64 v114; // [rsp+A8h] [rbp-238h]
  __int64 v115; // [rsp+B0h] [rbp-230h]
  __int64 v116[2]; // [rsp+C0h] [rbp-220h] BYREF
  __int64 v117; // [rsp+D0h] [rbp-210h]
  __m128i v118; // [rsp+E0h] [rbp-200h] BYREF
  __int64 v119; // [rsp+F0h] [rbp-1F0h] BYREF
  char *v120; // [rsp+F8h] [rbp-1E8h]
  __int16 v121; // [rsp+100h] [rbp-1E0h]
  char v122; // [rsp+108h] [rbp-1D8h] BYREF
  __int64 v123; // [rsp+110h] [rbp-1D0h]
  __int64 v124; // [rsp+118h] [rbp-1C8h]
  __int16 v125; // [rsp+120h] [rbp-1C0h]
  __int64 v126; // [rsp+128h] [rbp-1B8h]
  void **v127; // [rsp+130h] [rbp-1B0h]
  _QWORD *v128; // [rsp+138h] [rbp-1A8h]
  __int64 v129; // [rsp+140h] [rbp-1A0h]
  int v130; // [rsp+148h] [rbp-198h]
  __int16 v131; // [rsp+14Ch] [rbp-194h]
  char v132; // [rsp+14Eh] [rbp-192h]
  __int64 v133; // [rsp+150h] [rbp-190h]
  __int64 v134; // [rsp+158h] [rbp-188h]
  void *v135; // [rsp+160h] [rbp-180h] BYREF
  _QWORD v136[4]; // [rsp+168h] [rbp-178h] BYREF
  __int64 v137[8]; // [rsp+188h] [rbp-158h] BYREF
  _QWORD *v138; // [rsp+1C8h] [rbp-118h]
  unsigned int v139; // [rsp+1D8h] [rbp-108h]
  unsigned __int64 v140; // [rsp+1E8h] [rbp-F8h]
  char v141; // [rsp+1FCh] [rbp-E4h]
  unsigned __int64 v142; // [rsp+258h] [rbp-88h]
  char v143; // [rsp+26Ch] [rbp-74h]

  v94 = (_QWORD *)(a1 + 72);
  v102 = sub_25096F0((_QWORD *)(a1 + 72));
  v3 = (__int64 *)sub_2555710(*(_QWORD *)(*(_QWORD *)(a2 + 208) + 240LL), v102, 0);
  v4 = *(_QWORD *)(a1 + 136);
  v110 = v3;
  v105 = v4;
  v96 = v4 + 16LL * *(unsigned int *)(a1 + 144);
  result = 1;
  if ( v4 != v96 )
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(v105 + 8);
      v108 = v6;
      if ( *(_DWORD *)(v6 + 12) != 2 )
        break;
LABEL_97:
      v105 += 16;
      if ( v96 == v105 )
        return result;
    }
    v7 = *(_QWORD **)(v6 + 56);
    v106 = &v7[*(unsigned int *)(v6 + 64)];
    if ( v7 != v106 )
    {
      v8 = *(_QWORD **)(v6 + 56);
      v99 = a2 + 3880;
      v97 = a2 + 3912;
      do
      {
        v16 = *v8;
        v113 = 4;
        v114 = 0;
        v115 = v16;
        if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
          sub_BD73F0((__int64)&v113);
        if ( !*(_DWORD *)(a2 + 3896) )
        {
          v9 = *(__int64 **)(a2 + 3912);
          v10 = sub_2538140(v9, (__int64)&v9[3 * *(unsigned int *)(a2 + 3920)], (__int64)&v113);
          if ( v15 != v10 )
            goto LABEL_6;
          v30 = v13 + 1;
          v31 = &v113;
          if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 3924) )
          {
            if ( v9 > &v113 || v15 <= (unsigned __int64 *)&v113 )
            {
              sub_D6B130(v97, v30, v11, v12, v13, v14);
              LODWORD(v14) = *(_DWORD *)(a2 + 3920);
              v15 = (unsigned __int64 *)(*(_QWORD *)(a2 + 3912) + 24LL * (unsigned int)v14);
            }
            else
            {
              sub_D6B130(v97, v30, v11, v12, v13, v14);
              v89 = *(_QWORD *)(a2 + 3912);
              v14 = *(unsigned int *)(a2 + 3920);
              v31 = (__int64 *)(v89 + (char *)&v113 - (char *)v9);
              v15 = (unsigned __int64 *)(v89 + 24 * v14);
            }
          }
          if ( v15 )
          {
            *v15 = 4;
            v32 = v31[2];
            v15[1] = 0;
            v15[2] = v32;
            if ( v32 != 0 && v32 != -4096 && v32 != -8192 )
              sub_BD6050(v15, *v31 & 0xFFFFFFFFFFFFFFF8LL);
            LODWORD(v14) = *(_DWORD *)(a2 + 3920);
          }
          v33 = (unsigned int)(v14 + 1);
          *(_DWORD *)(a2 + 3920) = v33;
          if ( (unsigned int)v33 <= 8 )
            goto LABEL_6;
          v34 = *(_QWORD **)(a2 + 3912);
          v93 = v8;
          v35 = &v34[3 * v33];
          while ( 2 )
          {
            v36 = *(_DWORD *)(a2 + 3904);
            if ( !v36 )
            {
              ++*(_QWORD *)(a2 + 3880);
              v116[0] = 0;
              goto LABEL_38;
            }
            v41 = v34[2];
            v42 = *(_QWORD *)(a2 + 3888);
            v43 = (v36 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
            v44 = (unsigned __int64 *)(v42 + 24LL * v43);
            v45 = v44[2];
            if ( v45 != v41 )
            {
              v46 = 1;
              v37 = 0;
              while ( v45 != -4096 )
              {
                if ( v37 || v45 != -8192 )
                  v44 = v37;
                v43 = (v36 - 1) & (v46 + v43);
                v45 = *(_QWORD *)(v42 + 24LL * v43 + 16);
                if ( v41 == v45 )
                  goto LABEL_49;
                ++v46;
                v37 = v44;
                v44 = (unsigned __int64 *)(v42 + 24LL * v43);
              }
              v47 = *(_DWORD *)(a2 + 3896);
              if ( !v37 )
                v37 = v44;
              ++*(_QWORD *)(a2 + 3880);
              v38 = v47 + 1;
              v116[0] = (__int64)v37;
              if ( 4 * (v47 + 1) >= 3 * v36 )
              {
LABEL_38:
                v36 *= 2;
                goto LABEL_39;
              }
              if ( v36 - *(_DWORD *)(a2 + 3900) - v38 <= v36 >> 3 )
              {
LABEL_39:
                sub_2517BE0(v99, v36);
                sub_25116B0(v99, (__int64)v34, v116);
                v37 = (unsigned __int64 *)v116[0];
                v38 = *(_DWORD *)(a2 + 3896) + 1;
              }
              *(_DWORD *)(a2 + 3896) = v38;
              v118 = (__m128i)4uLL;
              v119 = -4096;
              if ( v37[2] != -4096 )
                --*(_DWORD *)(a2 + 3900);
              sub_D68D70(&v118);
              v39 = v37[2];
              v40 = v34[2];
              if ( v39 != v40 )
              {
                if ( v39 != 0 && v39 != -4096 && v39 != -8192 )
                {
                  sub_BD60C0(v37);
                  v40 = v34[2];
                }
                v37[2] = v40;
                if ( v40 != 0 && v40 != -4096 && v40 != -8192 )
                  sub_BD6050(v37, *v34 & 0xFFFFFFFFFFFFFFF8LL);
              }
            }
LABEL_49:
            v34 += 3;
            if ( v35 == v34 )
            {
              v8 = v93;
              goto LABEL_6;
            }
            continue;
          }
        }
        v17 = *(_DWORD *)(a2 + 3904);
        if ( v17 )
        {
          v77 = v17 - 1;
          v78 = *(_QWORD *)(a2 + 3888);
          v116[0] = 4;
          v116[1] = 0;
          v117 = -4096;
          v118 = (__m128i)4uLL;
          v119 = -8192;
          v79 = v77 & (((unsigned int)v115 >> 9) ^ ((unsigned int)v115 >> 4));
          v80 = (unsigned __int64 *)(v78 + 24LL * v79);
          v81 = v80[2];
          if ( v81 == v115 )
          {
LABEL_100:
            sub_D68D70(&v118);
            sub_D68D70(v116);
            goto LABEL_6;
          }
          v90 = 1;
          v18 = 0;
          while ( v81 != -4096 )
          {
            if ( v81 != -8192 || v18 )
              v80 = v18;
            v79 = v77 & (v90 + v79);
            v81 = *(_QWORD *)(v78 + 24LL * v79 + 16);
            if ( v115 == v81 )
              goto LABEL_100;
            v18 = v80;
            ++v90;
            v80 = (unsigned __int64 *)(v78 + 24LL * v79);
          }
          if ( !v18 )
            v18 = v80;
          sub_D68D70(&v118);
          sub_D68D70(v116);
          v17 = *(_DWORD *)(a2 + 3904);
          v19 = *(_DWORD *)(a2 + 3896) + 1;
          ++*(_QWORD *)(a2 + 3880);
          v116[0] = (__int64)v18;
          if ( 4 * v19 < 3 * v17 )
          {
            if ( v17 - *(_DWORD *)(a2 + 3900) - v19 > v17 >> 3 )
              goto LABEL_18;
            goto LABEL_17;
          }
        }
        else
        {
          ++*(_QWORD *)(a2 + 3880);
          v116[0] = 0;
        }
        v17 *= 2;
LABEL_17:
        sub_2517BE0(v99, v17);
        sub_25116B0(v99, (__int64)&v113, v116);
        v18 = (unsigned __int64 *)v116[0];
        v19 = *(_DWORD *)(a2 + 3896) + 1;
LABEL_18:
        *(_DWORD *)(a2 + 3896) = v19;
        v118 = (__m128i)4uLL;
        v119 = -4096;
        if ( v18[2] != -4096 )
          --*(_DWORD *)(a2 + 3900);
        sub_D68D70(&v118);
        sub_2538AB0(v18, &v113);
        v22 = *(unsigned int *)(a2 + 3920);
        v23 = *(unsigned int *)(a2 + 3924);
        v24 = v22 + 1;
        v25 = *(_DWORD *)(a2 + 3920);
        if ( v22 + 1 > v23 )
        {
          v91 = *(_QWORD *)(a2 + 3912);
          if ( v91 > (unsigned __int64)&v113 || (unsigned __int64)&v113 >= v91 + 24 * v22 )
          {
            sub_D6B130(v97, v24, v22, v23, v20, v21);
            v22 = *(unsigned int *)(a2 + 3920);
            v26 = *(_QWORD *)(a2 + 3912);
            v27 = &v113;
            v25 = *(_DWORD *)(a2 + 3920);
          }
          else
          {
            sub_D6B130(v97, v24, v22, v23, v20, v21);
            v26 = *(_QWORD *)(a2 + 3912);
            v22 = *(unsigned int *)(a2 + 3920);
            v27 = (__int64 *)((char *)&v113 + v26 - v91);
            v25 = *(_DWORD *)(a2 + 3920);
          }
        }
        else
        {
          v26 = *(_QWORD *)(a2 + 3912);
          v27 = &v113;
        }
        v28 = (unsigned __int64 *)(v26 + 24 * v22);
        if ( v28 )
        {
          *v28 = 4;
          v29 = v27[2];
          v28[1] = 0;
          v28[2] = v29;
          if ( v29 != 0 && v29 != -4096 && v29 != -8192 )
            sub_BD6050(v28, *v27 & 0xFFFFFFFFFFFFFFF8LL);
          v25 = *(_DWORD *)(a2 + 3920);
        }
        *(_DWORD *)(a2 + 3920) = v25 + 1;
LABEL_6:
        if ( v115 != 0 && v115 != -4096 && v115 != -8192 )
          sub_BD60C0(&v113);
        ++v8;
      }
      while ( v106 != v8 );
    }
    v113 = (__int64)&v110;
    v114 = v108;
    if ( *(_DWORD *)(v108 + 8) == 109 )
      sub_25600F0((_QWORD *)a2, *(_QWORD *)v108, "OMP110", 6u, (__int64)&v113);
    else
      sub_25600F0((_QWORD *)a2, *(_QWORD *)v108, "HeapToStack", 0xBu, (__int64)&v113);
    v48 = *(_QWORD *)(*(_QWORD *)(a2 + 208) + 104LL);
    v118.m128i_i64[0] = a2;
    v118.m128i_i64[1] = a1;
    v49 = sub_25096F0(v94);
    v50 = (__int64 *)sub_2555710(*(_QWORD *)(*(_QWORD *)(a2 + 208) + 240LL), v49, 0);
    sub_D5CDD0(
      (__int64)v116,
      *(_QWORD *)v108,
      v50,
      (__int64 (__fastcall *)(__int64, _QWORD))sub_254E230,
      (__int64)&v118);
    if ( (_BYTE)v117 )
    {
      v51 = (__int64 *)sub_BD5C60(*(_QWORD *)v108);
      v107 = sub_ACCFD0(v51, (__int64)v116);
    }
    else
    {
      v82 = sub_BD5C60(*(_QWORD *)v108);
      sub_D5EB90((__int64)&v118, v48, (__int64)v110, v82, 0, 0);
      v107 = sub_D63C20((__int64)&v118, *(_QWORD *)v108);
      if ( !v143 )
        _libc_free(v142);
      if ( !v141 )
        _libc_free(v140);
      v83 = v139;
      if ( v139 )
      {
        v84 = v138;
        v85 = &v138[7 * v139];
        do
        {
          if ( *v84 != -8192 && *v84 != -4096 )
          {
            sub_D68D70(v84 + 4);
            sub_D68D70(v84 + 1);
          }
          v84 += 7;
        }
        while ( v85 != v84 );
        v83 = v139;
      }
      sub_C7D6A0((__int64)v138, 56 * v83, 8);
      sub_B32BF0(v137);
      v136[2] = &unk_49D94D0;
      nullsub_63();
      if ( v120 != &v122 )
        _libc_free((unsigned __int64)v120);
    }
    if ( *(_BYTE *)(v108 + 17) )
    {
      v52 = *(_QWORD *)(v102 + 80);
      if ( !v52 )
        BUG();
      v53 = *(_QWORD *)v108;
      v98 = 1;
      v103 = *(_QWORD *)(v52 + 32);
    }
    else
    {
      v98 = 0;
      v53 = *(_QWORD *)v108;
      v103 = *(_QWORD *)v108 + 24LL;
    }
    v54 = sub_A74820((_QWORD *)(v53 + 72));
    v55 = v54;
    if ( !HIBYTE(v54) )
    {
      v56 = *(_QWORD *)(v53 - 32);
      if ( !v56 || (v55 = *(_BYTE *)v56) != 0 )
      {
        v55 = 0;
      }
      else if ( *(_QWORD *)(v56 + 24) == *(_QWORD *)(v53 + 80) )
      {
        v118.m128i_i64[0] = *(_QWORD *)(v56 + 120);
        v57 = sub_A74820(&v118);
        if ( HIBYTE(v57) )
          v55 = v57;
      }
    }
    v58 = sub_D5CD40(*(_QWORD *)v108, v110);
    if ( v58 )
    {
      v109 = 0;
      v59.m128i_i64[0] = sub_250D2C0(v58, 0);
      v118 = v59;
      v60 = sub_2527570(a2, &v118, a1, &v109);
      v112 = v61;
      v62 = v61;
      v111 = v60;
      if ( (_BYTE)v61 )
      {
        if ( v111 && *v111 == 17 )
        {
          v63 = *((_DWORD *)v111 + 8);
          v118.m128i_i32[2] = v63;
          if ( v63 > 0x40 )
          {
            sub_C43780((__int64)&v118, (const void **)v111 + 3);
            v63 = v118.m128i_u32[2];
          }
          else
          {
            v118.m128i_i64[0] = *((_QWORD *)v111 + 3);
          }
          LOBYTE(v119) = 1;
          v64 = v118.m128i_i64[0];
        }
        else
        {
          LOBYTE(v119) = 0;
          v63 = v118.m128i_u32[2];
          v62 = 0;
          v64 = v118.m128i_i64[0];
        }
        if ( v63 > 0x40 )
          v64 = *(_QWORD *)v64;
      }
      else
      {
        v118.m128i_i32[2] = 64;
        v64 = 0;
        v62 = 1;
        v118.m128i_i64[0] = 0;
      }
      if ( v64 )
      {
        _BitScanReverse64(&v64, v64);
        if ( v55 < (unsigned __int8)(63 - (v64 ^ 0x3F)) )
          v55 = 63 - (v64 ^ 0x3F);
      }
      if ( v62 )
      {
        LOBYTE(v119) = 0;
        sub_969240(v118.m128i_i64);
      }
    }
    v100 = *(_DWORD *)(v48 + 4);
    v65 = (_QWORD *)sub_B2BE50(v102);
    v66 = (__int64 *)sub_BCB2B0(v65);
    v118.m128i_i64[0] = (__int64)sub_BD5D20(*(_QWORD *)v108);
    v119 = (__int64)".h2s";
    v118.m128i_i64[1] = v67;
    v121 = 773;
    v68 = sub_BD2C40(80, unk_3F10A14);
    v69 = (__int64)v68;
    if ( v68 )
      sub_B4CCA0((__int64)v68, v66, v100, v107, v55, (__int64)&v118, v103, v98);
    v70 = *(_QWORD *)v108;
    if ( *(_QWORD *)(*(_QWORD *)v108 + 8LL) != *(_QWORD *)(v69 + 8) )
    {
      v118.m128i_i64[0] = (__int64)"malloc_cast";
      v121 = 259;
      v69 = sub_B52190(v69, *(_QWORD *)(*(_QWORD *)v108 + 8LL), (__int64)&v118, v70 + 24, 0);
    }
    v71 = (_QWORD *)sub_B2BE50(v102);
    v72 = (__int64 **)sub_BCB2B0(v71);
    v73 = (unsigned __int8 *)sub_D5D1D0(*(unsigned __int8 **)v108, v110, v72);
    sub_250D230((unsigned __int64 *)&v118, *(_QWORD *)v108, 1, 0);
    sub_256F570(a2, v118.m128i_i64[0], v118.m128i_i64[1], (unsigned __int8 *)v69, 1u);
    v74 = *(_QWORD *)v108;
    if ( **(_BYTE **)v108 == 34 )
    {
      v101 = *(_QWORD *)(v74 - 96);
      sub_B43C20((__int64)&v118, *(_QWORD *)(v74 + 40));
      v75 = v118.m128i_u16[4];
      v104 = v118.m128i_i64[0];
      v76 = sub_BD2C40(72, 1u);
      if ( v76 )
        sub_B4C8F0((__int64)v76, v101, 1u, v104, v75);
      sub_2570110(a2, *(_QWORD *)v108);
      if ( (unsigned int)*v73 - 12 <= 1 )
        goto LABEL_94;
    }
    else
    {
      sub_2570110(a2, v74);
      if ( (unsigned int)*v73 - 12 <= 1 )
      {
LABEL_94:
        if ( (_BYTE)v117 )
        {
          LOBYTE(v117) = 0;
          sub_969240(v116);
        }
        result = 0;
        goto LABEL_97;
      }
    }
    v86 = *(_QWORD *)(v69 + 32);
    if ( v86 == *(_QWORD *)(v69 + 40) + 48LL || !v86 )
      v87 = 0;
    else
      v87 = v86 - 24;
    v126 = sub_BD5C60(v87);
    v127 = &v135;
    v131 = 512;
    v118.m128i_i64[1] = 0x200000000LL;
    v128 = v136;
    v135 = &unk_49DA100;
    v125 = 0;
    v136[0] = &unk_49DA0B0;
    v118.m128i_i64[0] = (__int64)&v119;
    v129 = 0;
    v130 = 0;
    v132 = 7;
    v133 = 0;
    v134 = 0;
    v123 = 0;
    v124 = 0;
    sub_D5F1F0((__int64)&v118, v87);
    v88 = v92;
    BYTE1(v88) = 0;
    v92 = (unsigned __int8)v92;
    sub_B34240((__int64)&v118, v69, (__int64)v73, v107, v88, 0, 0, 0, 0);
    nullsub_61();
    v135 = &unk_49DA100;
    nullsub_63();
    if ( (__int64 *)v118.m128i_i64[0] != &v119 )
      _libc_free(v118.m128i_u64[0]);
    goto LABEL_94;
  }
  return result;
}
