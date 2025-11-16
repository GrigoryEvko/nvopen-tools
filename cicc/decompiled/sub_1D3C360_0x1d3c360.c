// Function: sub_1D3C360
// Address: 0x1d3c360
//
__int64 __fastcall sub_1D3C360(__int64 *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v7; // rsi
  __int64 v8; // r15
  __int64 v9; // rax
  __m128i v10; // xmm0
  char *v11; // rdx
  char v12; // cl
  __int64 v13; // rbx
  __int64 v14; // r8
  __int64 v15; // rax
  _QWORD *v16; // rbx
  int v17; // eax
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  unsigned int v21; // edx
  unsigned __int8 v22; // al
  unsigned __int64 v23; // rdx
  __int64 v24; // rcx
  unsigned int v25; // eax
  __int64 v26; // r15
  __int64 v27; // rbx
  unsigned __int8 *v28; // rax
  __int64 v29; // r14
  unsigned int v30; // eax
  __int64 v31; // rsi
  const void **v32; // r8
  __int64 v33; // r10
  __int64 v34; // rcx
  unsigned __int64 v35; // r9
  __int64 v36; // r15
  __int128 v37; // rax
  unsigned int v38; // edx
  __int64 v39; // r14
  __int128 v40; // rax
  unsigned int v41; // edx
  __int64 v42; // rax
  __int128 v43; // rax
  __int64 *v44; // rax
  __int64 v45; // r8
  unsigned int v46; // edx
  int v47; // eax
  unsigned __int64 v48; // rbx
  __int64 v49; // rax
  __int64 v50; // rax
  unsigned int v51; // edx
  __int64 v52; // r14
  __int64 v54; // rax
  unsigned int v55; // eax
  __int64 v56; // rdx
  __int64 v57; // rsi
  unsigned __int64 v58; // r11
  unsigned int v59; // esi
  int v60; // eax
  _QWORD *v61; // rax
  __int64 v62; // rax
  unsigned int v63; // esi
  __int64 v64; // rax
  unsigned __int64 v65; // rax
  const void **v66; // [rsp+8h] [rbp-138h]
  const void **v67; // [rsp+10h] [rbp-130h]
  __int64 v68; // [rsp+10h] [rbp-130h]
  const void **v69; // [rsp+18h] [rbp-128h]
  __int64 v70; // [rsp+18h] [rbp-128h]
  __int64 v71; // [rsp+18h] [rbp-128h]
  __int64 v72; // [rsp+20h] [rbp-120h]
  __int64 v73; // [rsp+20h] [rbp-120h]
  unsigned __int64 v74; // [rsp+20h] [rbp-120h]
  __int64 v75; // [rsp+28h] [rbp-118h]
  const void **v76; // [rsp+28h] [rbp-118h]
  const void **v77; // [rsp+28h] [rbp-118h]
  unsigned __int64 v78; // [rsp+28h] [rbp-118h]
  __int64 v79; // [rsp+28h] [rbp-118h]
  const void **v80; // [rsp+30h] [rbp-110h]
  __int64 v81; // [rsp+30h] [rbp-110h]
  unsigned __int64 v82; // [rsp+30h] [rbp-110h]
  unsigned __int64 v83; // [rsp+30h] [rbp-110h]
  __int64 v84; // [rsp+30h] [rbp-110h]
  __int64 v85; // [rsp+38h] [rbp-108h]
  __int64 v86; // [rsp+38h] [rbp-108h]
  __int64 v87; // [rsp+38h] [rbp-108h]
  __int64 v88; // [rsp+38h] [rbp-108h]
  __int64 v89; // [rsp+38h] [rbp-108h]
  __int64 v90; // [rsp+38h] [rbp-108h]
  __int64 v91; // [rsp+40h] [rbp-100h]
  __int64 v92; // [rsp+40h] [rbp-100h]
  __int64 v93; // [rsp+40h] [rbp-100h]
  __int64 v94; // [rsp+40h] [rbp-100h]
  __int64 v95; // [rsp+50h] [rbp-F0h]
  __int64 v96; // [rsp+58h] [rbp-E8h]
  __int64 *v97; // [rsp+70h] [rbp-D0h]
  __int64 *v98; // [rsp+78h] [rbp-C8h]
  __int64 v99; // [rsp+80h] [rbp-C0h]
  __int64 v100; // [rsp+80h] [rbp-C0h]
  __int64 *v101; // [rsp+80h] [rbp-C0h]
  unsigned __int64 v102; // [rsp+88h] [rbp-B8h]
  unsigned __int64 v103; // [rsp+88h] [rbp-B8h]
  __int64 v104; // [rsp+B0h] [rbp-90h] BYREF
  int v105; // [rsp+B8h] [rbp-88h]
  unsigned int v106; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v107; // [rsp+C8h] [rbp-78h]
  __int128 v108; // [rsp+D0h] [rbp-70h]
  __int64 v109; // [rsp+E0h] [rbp-60h]
  __int64 v110; // [rsp+F0h] [rbp-50h] BYREF
  __int64 v111; // [rsp+F8h] [rbp-48h]
  __int64 v112; // [rsp+100h] [rbp-40h]

  v7 = *(_QWORD *)(a2 + 72);
  v104 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v104, v7, 2);
  v8 = a1[2];
  v105 = *(_DWORD *)(a2 + 64);
  v9 = *(_QWORD *)(a2 + 32);
  v10 = _mm_loadu_si128((const __m128i *)(v9 + 40));
  v11 = *(char **)(a2 + 40);
  v12 = *v11;
  v97 = *(__int64 **)(*(_QWORD *)(v9 + 80) + 88LL);
  v107 = *((_QWORD *)v11 + 1);
  v13 = *(_QWORD *)(v9 + 8);
  LOBYTE(v106) = v12;
  v14 = *(_QWORD *)v9;
  v96 = v13;
  v15 = *(_QWORD *)(*(_QWORD *)(v9 + 120) + 88LL);
  v16 = *(_QWORD **)(v15 + 24);
  if ( *(_DWORD *)(v15 + 32) > 0x40u )
    v16 = (_QWORD *)*v16;
  v110 = 0;
  v17 = 0;
  v111 = 0;
  v112 = 0;
  v108 = (unsigned __int64)v97;
  LOBYTE(v109) = 0;
  if ( v97 )
  {
    v18 = *v97;
    if ( *(_BYTE *)(*v97 + 8) == 16 )
      v18 = **(_QWORD **)(v18 + 16);
    v17 = *(_DWORD *)(v18 + 8) >> 8;
  }
  v19 = a1[4];
  v99 = v14;
  HIDWORD(v109) = v17;
  v20 = sub_1E0A0C0(v19);
  v21 = 8 * sub_15A9520(v20, 0);
  if ( v21 == 32 )
  {
    v22 = 5;
  }
  else if ( v21 > 0x20 )
  {
    v22 = 6;
    if ( v21 != 64 )
    {
      v22 = 0;
      if ( v21 == 128 )
        v22 = 7;
    }
  }
  else
  {
    v22 = 3;
    if ( v21 != 8 )
      v22 = 4 * (v21 == 16);
  }
  v24 = sub_1D2B730(
          a1,
          v22,
          0,
          (__int64)&v104,
          v99,
          v96,
          v10.m128i_i64[0],
          v10.m128i_i64[1],
          v108,
          v109,
          0,
          0,
          (__int64)&v110,
          0);
  v95 = v24;
  v100 = v24;
  v25 = v23;
  v102 = v23;
  v98 = (__int64 *)v24;
  if ( (unsigned int)v16 > *(_DWORD *)(v8 + 84) )
  {
    v36 = 16LL * (unsigned int)v23;
    *(_QWORD *)&v37 = sub_1D38BB0(
                        (__int64)a1,
                        (unsigned int)((_DWORD)v16 - 1),
                        (__int64)&v104,
                        *(unsigned __int8 *)(v36 + *(_QWORD *)(v24 + 40)),
                        *(const void ***)(v36 + *(_QWORD *)(v24 + 40) + 8),
                        0,
                        v10,
                        a4,
                        a5,
                        0);
    v101 = sub_1D332F0(
             a1,
             52,
             (__int64)&v104,
             *(unsigned __int8 *)(*(_QWORD *)(v95 + 40) + v36),
             *(const void ***)(*(_QWORD *)(v95 + 40) + v36 + 8),
             0,
             *(double *)v10.m128i_i64,
             a4,
             a5,
             v100,
             v102,
             v37);
    v39 = 16LL * v38;
    v102 = v38 | v102 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v40 = sub_1D38BB0(
                        (__int64)a1,
                        -(__int64)(unsigned int)v16,
                        (__int64)&v104,
                        *(unsigned __int8 *)(v39 + v101[5]),
                        *(const void ***)(v39 + v101[5] + 8),
                        0,
                        v10,
                        a4,
                        a5,
                        0);
    v98 = sub_1D332F0(
            a1,
            118,
            (__int64)&v104,
            *(unsigned __int8 *)(v101[5] + v39),
            *(const void ***)(v101[5] + v39 + 8),
            0,
            *(double *)v10.m128i_i64,
            a4,
            a5,
            (__int64)v101,
            v102,
            v40);
    v25 = v41;
  }
  v26 = v25;
  v27 = 16LL * v25;
  v28 = (unsigned __int8 *)(v27 + v98[5]);
  v80 = (const void **)*((_QWORD *)v28 + 1);
  v85 = *v28;
  v29 = sub_1E0A0C0(a1[4]);
  v91 = sub_1F58E60(&v106, a1[6]);
  v30 = sub_15A9FE0(v29, v91);
  v31 = v91;
  v32 = v80;
  v33 = 1;
  v34 = v85;
  v35 = v30;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v31 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v54 = *(_QWORD *)(v31 + 32);
        v31 = *(_QWORD *)(v31 + 24);
        v33 *= v54;
        continue;
      case 1:
        v42 = 16;
        goto LABEL_22;
      case 2:
        v42 = 32;
        goto LABEL_22;
      case 3:
      case 9:
        v42 = 64;
        goto LABEL_22;
      case 4:
        v42 = 80;
        goto LABEL_22;
      case 5:
      case 6:
        v42 = 128;
        goto LABEL_22;
      case 7:
        v76 = v80;
        v59 = 0;
        v82 = v35;
        v87 = v33;
        v93 = v34;
        goto LABEL_35;
      case 0xB:
        v42 = *(_DWORD *)(v31 + 8) >> 8;
        goto LABEL_22;
      case 0xD:
        v77 = v80;
        v83 = v35;
        v88 = v33;
        v94 = v34;
        v61 = (_QWORD *)sub_15A9930(v29, v31);
        v34 = v94;
        v33 = v88;
        v35 = v83;
        v32 = v77;
        v42 = 8LL * *v61;
        goto LABEL_22;
      case 0xE:
        v69 = v80;
        v72 = v35;
        v75 = v33;
        v81 = v85;
        v86 = *(_QWORD *)(v31 + 24);
        v92 = *(_QWORD *)(v31 + 32);
        v55 = sub_15A9FE0(v29, v86);
        v32 = v69;
        v35 = v72;
        v56 = 1;
        v57 = v86;
        v33 = v75;
        v58 = v55;
        v34 = v81;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v57 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v64 = *(_QWORD *)(v57 + 32);
              v57 = *(_QWORD *)(v57 + 24);
              v56 *= v64;
              continue;
            case 1:
              v62 = 16;
              goto LABEL_43;
            case 2:
              v62 = 32;
              goto LABEL_43;
            case 3:
            case 9:
              v62 = 64;
              goto LABEL_43;
            case 4:
              v62 = 80;
              goto LABEL_43;
            case 5:
            case 6:
              v62 = 128;
              goto LABEL_43;
            case 7:
              v67 = v69;
              v63 = 0;
              v70 = v72;
              v73 = v75;
              v78 = v58;
              v84 = v56;
              v89 = v34;
              goto LABEL_49;
            case 0xB:
              v62 = *(_DWORD *)(v57 + 8) >> 8;
              goto LABEL_43;
            case 0xD:
              v67 = v69;
              v70 = v72;
              v73 = v75;
              v78 = v58;
              v84 = v56;
              v89 = v34;
              v62 = 8LL * *(_QWORD *)sub_15A9930(v29, v57);
              goto LABEL_50;
            case 0xE:
              v66 = v69;
              v68 = v72;
              v71 = v75;
              v74 = v58;
              v79 = v56;
              v90 = *(_QWORD *)(v57 + 32);
              v65 = sub_12BE0A0(v29, *(_QWORD *)(v57 + 24));
              v34 = v81;
              v56 = v79;
              v58 = v74;
              v33 = v71;
              v35 = v68;
              v32 = v66;
              v62 = 8 * v90 * v65;
              goto LABEL_43;
            case 0xF:
              v67 = v69;
              v70 = v72;
              v73 = v75;
              v63 = *(_DWORD *)(v57 + 8) >> 8;
              v78 = v58;
              v84 = v56;
              v89 = v34;
LABEL_49:
              v62 = 8 * (unsigned int)sub_15A9520(v29, v63);
LABEL_50:
              v34 = v89;
              v56 = v84;
              v58 = v78;
              v33 = v73;
              v35 = v70;
              v32 = v67;
LABEL_43:
              v42 = 8 * v92 * v58 * ((v58 + ((unsigned __int64)(v62 * v56 + 7) >> 3) - 1) / v58);
              break;
          }
          goto LABEL_22;
        }
      case 0xF:
        v76 = v80;
        v82 = v35;
        v87 = v33;
        v59 = *(_DWORD *)(v31 + 8) >> 8;
        v93 = v34;
LABEL_35:
        v60 = sub_15A9520(v29, v59);
        v34 = v93;
        v33 = v87;
        v35 = v82;
        v32 = v76;
        v42 = (unsigned int)(8 * v60);
LABEL_22:
        *(_QWORD *)&v43 = sub_1D38BB0(
                            (__int64)a1,
                            v35 * ((v35 + ((unsigned __int64)(v33 * v42 + 7) >> 3) - 1) / v35),
                            (__int64)&v104,
                            v34,
                            v32,
                            0,
                            v10,
                            a4,
                            a5,
                            0);
        v103 = v26 | v102 & 0xFFFFFFFF00000000LL;
        v44 = sub_1D332F0(
                a1,
                52,
                (__int64)&v104,
                *(unsigned __int8 *)(v98[5] + v27),
                *(const void ***)(v98[5] + v27 + 8),
                0,
                *(double *)v10.m128i_i64,
                a4,
                a5,
                (__int64)v98,
                v103,
                v43);
        v110 = 0;
        v45 = (__int64)v44;
        v47 = 0;
        v111 = 0;
        v112 = 0;
        v48 = v46 | v96 & 0xFFFFFFFF00000000LL;
        v108 = (unsigned __int64)v97;
        LOBYTE(v109) = 0;
        if ( v97 )
        {
          v49 = *v97;
          if ( *(_BYTE *)(*v97 + 8) == 16 )
            v49 = **(_QWORD **)(v49 + 16);
          v47 = *(_DWORD *)(v49 + 8) >> 8;
        }
        HIDWORD(v109) = v47;
        v50 = sub_1D2BF40(
                a1,
                v95,
                1,
                (__int64)&v104,
                v45,
                v48,
                v10.m128i_i64[0],
                v10.m128i_i64[1],
                v108,
                v109,
                0,
                0,
                (__int64)&v110);
        v109 = 0;
        v110 = 0;
        v111 = 0;
        v112 = 0;
        v108 = 0u;
        v52 = sub_1D2B730(
                a1,
                v106,
                v107,
                (__int64)&v104,
                v50,
                v48 & 0xFFFFFFFF00000000LL | v51,
                (__int64)v98,
                v103,
                0,
                0,
                0,
                0,
                (__int64)&v110,
                0);
        if ( v104 )
          sub_161E7C0((__int64)&v104, v104);
        return v52;
    }
  }
}
