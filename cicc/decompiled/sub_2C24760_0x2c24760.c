// Function: sub_2C24760
// Address: 0x2c24760
//
__int64 __fastcall sub_2C24760(_QWORD *a1, __int64 a2)
{
  int v4; // eax
  __int64 v5; // rsi
  __int64 v6; // rdi
  int v7; // ecx
  unsigned int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 result; // rax
  unsigned int v14; // edx
  _QWORD *v15; // rax
  int v16; // edx
  unsigned int v17; // esi
  unsigned int v18; // ecx
  __int64 v19; // rdx
  int v20; // r8d
  __int64 v21; // rax
  __int64 *v22; // rdi
  __m128i v23; // xmm1
  __m128i v24; // xmm0
  __m128i v25; // xmm2
  __m128i v26; // xmm3
  __m128i v27; // xmm4
  __m128i v28; // xmm5
  __int64 v29; // rdi
  __int64 v30; // rcx
  __int64 v31; // rax
  char v32; // al
  __int64 v33; // r8
  bool v34; // zf
  _QWORD *v35; // rax
  int v36; // esi
  int v37; // edx
  unsigned int v38; // esi
  __int64 v39; // rdx
  _QWORD *v40; // [rsp+8h] [rbp-468h]
  __int64 v41; // [rsp+10h] [rbp-460h]
  __int64 *v42; // [rsp+10h] [rbp-460h]
  __int64 v43; // [rsp+18h] [rbp-458h]
  __m128i v44; // [rsp+20h] [rbp-450h] BYREF
  void (__fastcall *v45)(__m128i *, __m128i *, __int64); // [rsp+30h] [rbp-440h]
  char (__fastcall *v46)(__int64 *, __int64 *); // [rsp+38h] [rbp-438h]
  _QWORD *v47; // [rsp+40h] [rbp-430h] BYREF
  __m128i v48; // [rsp+48h] [rbp-428h] BYREF
  __int64 (__fastcall *v49)(char *, __m128i *, int); // [rsp+58h] [rbp-418h]
  char (__fastcall *v50)(__int64 *, __int64 *); // [rsp+60h] [rbp-410h]
  _QWORD *v51; // [rsp+70h] [rbp-400h] BYREF
  void *v52; // [rsp+78h] [rbp-3F8h]
  __int64 v53; // [rsp+80h] [rbp-3F0h]
  __m128i v54; // [rsp+88h] [rbp-3E8h] BYREF
  __m128i v55; // [rsp+98h] [rbp-3D8h] BYREF
  __m128i v56; // [rsp+A8h] [rbp-3C8h] BYREF
  __m128i v57; // [rsp+B8h] [rbp-3B8h] BYREF
  __int64 v58; // [rsp+C8h] [rbp-3A8h]
  _QWORD v59[3]; // [rsp+D0h] [rbp-3A0h] BYREF
  char v60; // [rsp+E8h] [rbp-388h]
  __int64 v61; // [rsp+F0h] [rbp-380h]
  __int64 v62; // [rsp+F8h] [rbp-378h]
  __int64 v63; // [rsp+100h] [rbp-370h]
  int v64; // [rsp+108h] [rbp-368h]
  __int64 v65; // [rsp+110h] [rbp-360h]
  __int64 v66; // [rsp+118h] [rbp-358h]
  __int64 v67; // [rsp+120h] [rbp-350h]
  __int64 v68; // [rsp+128h] [rbp-348h]
  __int64 v69; // [rsp+130h] [rbp-340h]
  __int64 v70; // [rsp+138h] [rbp-338h]
  __int64 v71; // [rsp+140h] [rbp-330h]
  __int64 v72; // [rsp+148h] [rbp-328h]
  __int64 v73; // [rsp+150h] [rbp-320h]
  char *v74; // [rsp+158h] [rbp-318h]
  __int64 v75; // [rsp+160h] [rbp-310h]
  int v76; // [rsp+168h] [rbp-308h]
  char v77; // [rsp+16Ch] [rbp-304h]
  char v78; // [rsp+170h] [rbp-300h] BYREF
  __int64 v79; // [rsp+1F0h] [rbp-280h]
  __int64 v80; // [rsp+1F8h] [rbp-278h]
  __int64 v81; // [rsp+200h] [rbp-270h]
  int v82; // [rsp+208h] [rbp-268h]
  char *v83; // [rsp+210h] [rbp-260h]
  __int64 v84; // [rsp+218h] [rbp-258h]
  char v85; // [rsp+220h] [rbp-250h] BYREF
  __int64 v86; // [rsp+250h] [rbp-220h]
  __int64 v87; // [rsp+258h] [rbp-218h]
  __int64 v88; // [rsp+260h] [rbp-210h]
  int v89; // [rsp+268h] [rbp-208h]
  __int64 v90; // [rsp+270h] [rbp-200h]
  char *v91; // [rsp+278h] [rbp-1F8h]
  __int64 v92; // [rsp+280h] [rbp-1F0h]
  int v93; // [rsp+288h] [rbp-1E8h]
  char v94; // [rsp+28Ch] [rbp-1E4h]
  char v95; // [rsp+290h] [rbp-1E0h] BYREF
  __int64 v96; // [rsp+2A0h] [rbp-1D0h]
  __int64 v97; // [rsp+2A8h] [rbp-1C8h]
  __int64 v98; // [rsp+2B0h] [rbp-1C0h]
  __int64 v99; // [rsp+2B8h] [rbp-1B8h]
  __int64 v100; // [rsp+2C0h] [rbp-1B0h]
  __int64 v101; // [rsp+2C8h] [rbp-1A8h]
  __int16 v102; // [rsp+2D0h] [rbp-1A0h]
  char v103; // [rsp+2D2h] [rbp-19Eh]
  char *v104; // [rsp+2D8h] [rbp-198h]
  __int64 v105; // [rsp+2E0h] [rbp-190h]
  char v106; // [rsp+2E8h] [rbp-188h] BYREF
  __int64 v107; // [rsp+308h] [rbp-168h]
  __int64 v108; // [rsp+310h] [rbp-160h]
  __int16 v109; // [rsp+318h] [rbp-158h]
  __int64 v110; // [rsp+320h] [rbp-150h]
  _QWORD *v111; // [rsp+328h] [rbp-148h]
  void **v112; // [rsp+330h] [rbp-140h]
  __int64 v113; // [rsp+338h] [rbp-138h]
  int v114; // [rsp+340h] [rbp-130h]
  __int16 v115; // [rsp+344h] [rbp-12Ch]
  char v116; // [rsp+346h] [rbp-12Ah]
  __int64 v117; // [rsp+348h] [rbp-128h]
  __int64 v118; // [rsp+350h] [rbp-120h]
  _QWORD v119[3]; // [rsp+358h] [rbp-118h] BYREF
  __m128i v120; // [rsp+370h] [rbp-100h]
  __m128i v121; // [rsp+380h] [rbp-F0h]
  __m128i v122; // [rsp+390h] [rbp-E0h]
  __m128i v123; // [rsp+3A0h] [rbp-D0h]
  __int64 v124; // [rsp+3B0h] [rbp-C0h]
  void *v125; // [rsp+3B8h] [rbp-B8h] BYREF
  char v126[16]; // [rsp+3C0h] [rbp-B0h] BYREF
  __int64 (__fastcall *v127)(char *, __m128i *, int); // [rsp+3D0h] [rbp-A0h]
  char (__fastcall *v128)(__int64 *, __int64 *); // [rsp+3D8h] [rbp-98h]
  char *v129; // [rsp+3E0h] [rbp-90h]
  __int64 v130; // [rsp+3E8h] [rbp-88h]
  char v131; // [rsp+3F0h] [rbp-80h] BYREF
  const char *v132; // [rsp+430h] [rbp-40h]

  v4 = *(_DWORD *)(a2 + 968);
  v5 = a1[19];
  v6 = *(_QWORD *)(a2 + 952);
  if ( v4 )
  {
    v7 = v4 - 1;
    v8 = (v4 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v9 = *(_QWORD *)(v6 + 16LL * v8);
    if ( v5 == v9 )
    {
LABEL_3:
      v10 = *(_QWORD *)(a2 + 904);
      v47 = (_QWORD *)a1[10];
      if ( (unsigned __int8)sub_2ABFB80(a2 + 120, (__int64 *)&v47, &v51) )
      {
        v11 = v51 + 1;
LABEL_5:
        v12 = *v11;
        *(_WORD *)(v10 + 64) = 0;
        *(_QWORD *)(v10 + 48) = v12;
        result = v12 + 48;
        *(_QWORD *)(v10 + 56) = result;
        return result;
      }
      v14 = *(_DWORD *)(a2 + 128);
      v15 = v51;
      ++*(_QWORD *)(a2 + 120);
      v59[0] = v15;
      v16 = (v14 >> 1) + 1;
      if ( (*(_BYTE *)(a2 + 128) & 1) != 0 )
      {
        v18 = 12;
        v17 = 4;
      }
      else
      {
        v17 = *(_DWORD *)(a2 + 144);
        v18 = 3 * v17;
      }
      if ( 4 * v16 >= v18 )
      {
        v17 *= 2;
      }
      else if ( v17 - (v16 + *(_DWORD *)(a2 + 132)) > v17 >> 3 )
      {
LABEL_10:
        *(_DWORD *)(a2 + 128) = *(_DWORD *)(a2 + 128) & 1 | (2 * v16);
        if ( *v15 != -4096 )
          --*(_DWORD *)(a2 + 132);
        v19 = (__int64)v47;
        v11 = v15 + 1;
        *v11 = 0;
        *(v11 - 1) = v19;
        goto LABEL_5;
      }
      sub_2ACA3E0(a2 + 120, v17);
      sub_2ABFB80(a2 + 120, (__int64 *)&v47, v59);
      v15 = (_QWORD *)v59[0];
      v16 = (*(_DWORD *)(a2 + 128) >> 1) + 1;
      goto LABEL_10;
    }
    v20 = 1;
    while ( v9 != -4096 )
    {
      v8 = v7 & (v20 + v8);
      v9 = *(_QWORD *)(v6 + 16LL * v8);
      if ( v5 == v9 )
        goto LABEL_3;
      ++v20;
    }
  }
  v21 = sub_AA4E30(*(_QWORD *)(a2 + 104));
  v22 = (__int64 *)a1[20];
  v74 = &v78;
  v83 = &v85;
  v84 = 0x200000000LL;
  v59[0] = v22;
  v59[1] = v21;
  v59[2] = "induction";
  v60 = 1;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v75 = 16;
  v76 = 0;
  v77 = 1;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v91 = &v95;
  v23 = _mm_loadu_si128(&v48);
  v102 = 1;
  v51 = &unk_49E5698;
  v44.m128i_i64[0] = (__int64)v59;
  v49 = (__int64 (__fastcall *)(char *, __m128i *, int))sub_27BFDD0;
  v24 = _mm_loadu_si128(&v44);
  v53 = v21;
  v46 = v50;
  v44 = v23;
  v50 = sub_27BFD20;
  v48 = v24;
  v92 = 2;
  v93 = 0;
  v94 = 1;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v103 = 0;
  v47 = &unk_49DA0D8;
  v45 = 0;
  v52 = &unk_49D94D0;
  v54 = (__m128i)(unsigned __int64)v21;
  LOWORD(v58) = 257;
  v55 = 0u;
  v56 = 0u;
  v57 = 0u;
  v110 = sub_B2BE50(*v22);
  v25 = _mm_loadu_si128(&v54);
  v111 = v119;
  v26 = _mm_loadu_si128(&v55);
  v112 = &v125;
  v27 = _mm_loadu_si128(&v56);
  v28 = _mm_loadu_si128(&v57);
  v104 = &v106;
  v119[2] = v53;
  v105 = 0x200000000LL;
  v124 = v58;
  v113 = 0;
  v114 = 0;
  v115 = 512;
  v116 = 7;
  v117 = 0;
  v118 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v119[0] = &unk_49E5698;
  v119[1] = &unk_49D94D0;
  v125 = &unk_49DA0D8;
  v127 = 0;
  v120 = v25;
  v121 = v26;
  v122 = v27;
  v123 = v28;
  if ( v49 )
  {
    v49(v126, &v48, 2);
    v128 = v50;
    v127 = (__int64 (__fastcall *)(_QWORD *, _QWORD *, int))v49;
  }
  v51 = &unk_49E5698;
  v52 = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  sub_B32BF0(&v47);
  if ( v45 )
    v45(&v44, &v44, 3);
  v29 = a1[19];
  v129 = &v131;
  v130 = 0x800000000LL;
  v132 = byte_3F871B3;
  v30 = *(_QWORD *)(*(_QWORD *)(a2 + 904) + 56LL);
  if ( v30 )
    v30 -= 24;
  v43 = a2 + 944;
  v41 = v30;
  v31 = sub_D95540(v29);
  v40 = sub_F8DB90((__int64)v59, a1[19], v31, v41 + 24, 0);
  v32 = sub_2C1BC00(a2 + 944, a1 + 19, &v47);
  v33 = (__int64)v40;
  v34 = v32 == 0;
  v35 = v47;
  if ( v34 )
  {
    v36 = *(_DWORD *)(a2 + 960);
    v51 = v47;
    ++*(_QWORD *)(a2 + 944);
    v37 = v36 + 1;
    v38 = *(_DWORD *)(a2 + 968);
    if ( 4 * v37 >= 3 * v38 )
    {
      v38 *= 2;
      v42 = a1 + 19;
    }
    else
    {
      if ( v38 - *(_DWORD *)(a2 + 964) - v37 > v38 >> 3 )
      {
LABEL_24:
        *(_DWORD *)(a2 + 960) = v37;
        if ( *v35 != -4096 )
          --*(_DWORD *)(a2 + 964);
        v39 = a1[19];
        v35[1] = 0;
        *v35 = v39;
        goto LABEL_27;
      }
      v42 = a1 + 19;
    }
    sub_2C24580(v43, v38);
    sub_2C1BC00(v43, v42, &v51);
    v33 = (__int64)v40;
    v37 = *(_DWORD *)(a2 + 960) + 1;
    v35 = v51;
    goto LABEL_24;
  }
LABEL_27:
  v35[1] = v33;
  LODWORD(v51) = 0;
  BYTE4(v51) = 0;
  sub_2AC6E90(a2, (__int64)(a1 + 12), v33, (unsigned int *)&v51);
  return sub_27C20B0((__int64)v59);
}
