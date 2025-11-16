// Function: sub_3055900
// Address: 0x3055900
//
__int64 __fastcall sub_3055900(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int16 v15; // ax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned int v19; // edx
  __int64 v20; // rax
  int v21; // ecx
  int v22; // ecx
  unsigned __int16 v23; // ax
  int v24; // r14d
  __int64 v25; // r8
  __int64 *v26; // rdi
  const __m128i *v27; // rax
  __m128i v28; // xmm1
  __int64 v29; // rax
  int v30; // edx
  unsigned __int16 v31; // ax
  __int64 v32; // rsi
  __int32 v33; // ecx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r9
  __int64 v37; // rdx
  __int64 v38; // r8
  __int64 *v39; // rdx
  unsigned __int64 v40; // rcx
  __int64 v41; // rax
  unsigned int i; // r14d
  __m128i v43; // xmm0
  int v44; // eax
  int v45; // edx
  int v46; // eax
  unsigned __int64 v47; // r14
  __int64 v48; // r10
  __int64 v49; // r11
  int v50; // r9d
  __int64 v51; // rsi
  __int64 *v52; // rcx
  __int64 v53; // rax
  unsigned int v54; // edx
  unsigned int v55; // r14d
  __int64 v56; // r11
  __int16 *v57; // rax
  __int64 v58; // r11
  unsigned __int16 v59; // dx
  __int64 (__fastcall *v60)(__int64, __int64, unsigned int); // r9
  __int64 v61; // rax
  __int64 v62; // rsi
  unsigned __int64 v63; // r8
  __int64 v64; // rax
  __int64 v65; // rdi
  int v66; // r14d
  char v67; // al
  unsigned __int64 v68; // rsi
  int v69; // eax
  __int64 v70; // rcx
  __int16 v71; // dx
  __int64 *v72; // r8
  __int64 *v73; // r11
  __int64 v74; // rax
  unsigned int v75; // edx
  __int64 v76; // [rsp+0h] [rbp-200h]
  __int64 v77; // [rsp+8h] [rbp-1F8h]
  unsigned __int8 v78; // [rsp+17h] [rbp-1E9h]
  __int64 *v79; // [rsp+18h] [rbp-1E8h]
  __int64 *v80; // [rsp+18h] [rbp-1E8h]
  int v81; // [rsp+20h] [rbp-1E0h]
  __int16 v82; // [rsp+20h] [rbp-1E0h]
  int v83; // [rsp+28h] [rbp-1D8h]
  __int64 *v84; // [rsp+28h] [rbp-1D8h]
  __m128i v85; // [rsp+30h] [rbp-1D0h] BYREF
  unsigned __int64 v86; // [rsp+40h] [rbp-1C0h]
  __int64 (__fastcall *v87)(__int64, __int64, unsigned int); // [rsp+48h] [rbp-1B8h]
  __int64 v88; // [rsp+50h] [rbp-1B0h]
  __int64 v89; // [rsp+58h] [rbp-1A8h]
  unsigned __int64 v90; // [rsp+60h] [rbp-1A0h] BYREF
  char *v91; // [rsp+68h] [rbp-198h] BYREF
  unsigned __int16 v92; // [rsp+70h] [rbp-190h] BYREF
  __int64 v93; // [rsp+78h] [rbp-188h]
  __int64 v94; // [rsp+80h] [rbp-180h] BYREF
  int v95; // [rsp+88h] [rbp-178h]
  __int64 v96; // [rsp+90h] [rbp-170h]
  __int64 v97; // [rsp+98h] [rbp-168h]
  __int64 v98; // [rsp+A0h] [rbp-160h] BYREF
  __int64 v99; // [rsp+A8h] [rbp-158h]
  __int64 v100; // [rsp+B0h] [rbp-150h]
  __int64 v101; // [rsp+B8h] [rbp-148h]
  unsigned __int64 v102[2]; // [rsp+C0h] [rbp-140h] BYREF
  _BYTE v103[72]; // [rsp+D0h] [rbp-130h] BYREF
  int v104; // [rsp+118h] [rbp-E8h] BYREF
  unsigned __int64 v105; // [rsp+120h] [rbp-E0h]
  int *v106; // [rsp+128h] [rbp-D8h]
  int *v107; // [rsp+130h] [rbp-D0h]
  __int64 v108; // [rsp+138h] [rbp-C8h]
  _OWORD *v109; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v110; // [rsp+148h] [rbp-B8h]
  _OWORD v111[11]; // [rsp+150h] [rbp-B0h] BYREF

  v102[1] = 0x400000000LL;
  v90 = 0;
  v91 = 0;
  v102[0] = (unsigned __int64)v103;
  v104 = 0;
  v105 = 0;
  v106 = &v104;
  v107 = &v104;
  v108 = 0;
  if ( !sub_3055560(a1, &v90, &v91, (__int64)v102, a6) )
    goto LABEL_4;
  v8 = *(unsigned __int16 *)(v90 + 96);
  v9 = *(_QWORD *)(v90 + 104);
  v92 = v8;
  v93 = v9;
  if ( (_WORD)v8 )
  {
    if ( (unsigned __int16)(v8 - 17) > 0xD3u )
    {
LABEL_4:
      v10 = 0;
      goto LABEL_5;
    }
    v15 = word_4456580[v8 - 1];
    v16 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v92) )
      goto LABEL_4;
    v15 = sub_3009970((__int64)&v92, 0, v12, v13, v14);
  }
  LOWORD(v109) = v15;
  v110 = v16;
  if ( v15 )
  {
    if ( (unsigned __int16)(v15 - 2) > 7u
      && (unsigned __int16)(v15 - 17) > 0x6Cu
      && (unsigned __int16)(v15 - 176) > 0x1Fu )
    {
      goto LABEL_4;
    }
  }
  else if ( !sub_3007070((__int64)&v109) )
  {
    goto LABEL_4;
  }
  if ( v92 )
  {
    if ( v92 == 1 || (unsigned __int16)(v92 - 504) <= 7u )
      BUG();
    v18 = 16LL * (v92 - 1);
    v17 = *(_QWORD *)&byte_444C4A0[v18];
    LOBYTE(v18) = byte_444C4A0[v18 + 8];
  }
  else
  {
    v17 = sub_3007260((__int64)&v92);
    v96 = v17;
    v97 = v18;
  }
  LOBYTE(v110) = v18;
  v109 = (_OWORD *)v17;
  v19 = sub_CA1930(&v109);
  if ( v91 != (char *)v19 || ((v19 - 16) & 0xFFFFFFEF) != 0 && v19 != 64 )
    goto LABEL_4;
  v20 = *(_QWORD *)(v90 + 56);
  if ( !v20 )
    goto LABEL_4;
  v21 = 1;
  do
  {
    if ( !*(_DWORD *)(v20 + 8) )
    {
      if ( !v21 )
        goto LABEL_4;
      v20 = *(_QWORD *)(v20 + 32);
      if ( !v20 )
        goto LABEL_34;
      if ( !*(_DWORD *)(v20 + 8) )
        goto LABEL_4;
      v21 = 0;
    }
    v20 = *(_QWORD *)(v20 + 32);
  }
  while ( v20 );
  if ( v21 == 1 )
    goto LABEL_4;
LABEL_34:
  v22 = *(_DWORD *)(v90 + 24);
  v23 = 2;
  if ( v19 != 1 )
  {
    v23 = 3;
    if ( v19 != 2 )
    {
      v23 = 4;
      if ( v19 != 4 )
      {
        v23 = 5;
        if ( v19 != 8 )
        {
          switch ( v19 )
          {
            case 0x10u:
              v23 = 6;
              break;
            case 0x20u:
              v23 = 7;
              break;
            case 0x40u:
              v23 = 8;
              break;
            default:
              v23 = 9 * (v19 == 128);
              break;
          }
        }
      }
    }
  }
  v89 = 0;
  v88 = v23;
  if ( v22 == 548 )
  {
    v109 = 0;
    v110 = 0;
    v111[0] = 0u;
    v65 = *(_QWORD *)(v90 + 112);
    v66 = *(unsigned __int16 *)(v65 + 32);
    v67 = sub_2EAC4F0(v65);
    v68 = v90;
    LOBYTE(v71) = v67;
    v69 = *(_DWORD *)(v90 + 24);
    v70 = *(_QWORD *)(v90 + 112);
    HIBYTE(v71) = 1;
    if ( v69 <= 365 )
    {
      if ( v69 <= 363 )
      {
        if ( v69 != 339 && (v69 & 0xFFFFFFBF) != 0x12B )
          goto LABEL_83;
LABEL_91:
        v72 = *(__int64 **)(v90 + 40);
        v73 = v72 + 10;
        goto LABEL_84;
      }
LABEL_89:
      v72 = *(__int64 **)(v90 + 40);
      v73 = v72 + 15;
      goto LABEL_84;
    }
    if ( v69 <= 467 )
    {
      if ( v69 > 464 )
        goto LABEL_91;
    }
    else if ( v69 == 497 )
    {
      goto LABEL_89;
    }
LABEL_83:
    v72 = *(__int64 **)(v90 + 40);
    v73 = v72 + 5;
LABEL_84:
    v74 = *(_QWORD *)(v90 + 80);
    v87 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))&v98;
    v98 = v74;
    if ( v74 )
    {
      v80 = v72;
      v82 = v71;
      v84 = v73;
      v85.m128i_i64[0] = v70;
      v86 = v90;
      sub_2AAAFA0(&v98);
      v72 = v80;
      v71 = v82;
      v73 = v84;
      v70 = v85.m128i_i64[0];
      v68 = v86;
    }
    LODWORD(v99) = *(_DWORD *)(v68 + 72);
    v88 = sub_33F1F00(
            a2,
            v88,
            v89,
            (_DWORD)v87,
            *v72,
            v72[1],
            *v73,
            v73[1],
            *(_OWORD *)v70,
            *(_QWORD *)(v70 + 16),
            v71,
            v66,
            (__int64)&v109,
            0);
    v55 = v75;
    sub_9C6650(v87);
    v56 = v88;
    goto LABEL_70;
  }
  v24 = 0;
  v25 = *(_QWORD *)(a2 + 16);
  if ( v22 == 551 )
    v24 = 8960;
  v26 = *(__int64 **)(a2 + 40);
  v109 = v111;
  v86 = (unsigned __int64)v111;
  v110 = 0x800000000LL;
  v27 = *(const __m128i **)(v90 + 40);
  v85.m128i_i64[0] = v25;
  v28 = _mm_loadu_si128(v27);
  LODWORD(v110) = 1;
  v111[0] = v28;
  v87 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v25 + 32LL);
  v29 = sub_2E79000(v26);
  if ( v87 == sub_2D42F30 )
  {
    v30 = sub_AE2980(v29, 0)[1];
    v31 = 2;
    if ( v30 != 1 )
    {
      v31 = 3;
      if ( v30 != 2 )
      {
        v31 = 4;
        if ( v30 != 4 )
        {
          switch ( v30 )
          {
            case 8:
              v31 = 5;
              break;
            case 16:
              v31 = 6;
              break;
            case 32:
              v31 = 7;
              break;
            case 64:
              v31 = 8;
              break;
            default:
              v31 = 9 * (v30 == 128);
              break;
          }
        }
      }
    }
  }
  else
  {
    v31 = v87(v85.m128i_i64[0], v29, 0);
  }
  v32 = *(_QWORD *)(a1 + 80);
  v87 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))&v98;
  v33 = v31;
  v98 = v32;
  if ( v32 )
  {
    v85.m128i_i64[0] = v31;
    sub_B96E90((__int64)&v98, v32, 1);
    v33 = v85.m128i_i32[0];
  }
  LODWORD(v99) = *(_DWORD *)(a1 + 72);
  v34 = sub_3400BD0(a2, v24, (_DWORD)v87, v33, 0, 0, 0);
  v36 = v35;
  v37 = (unsigned int)v110;
  v38 = v34;
  if ( (unsigned __int64)(unsigned int)v110 + 1 > HIDWORD(v110) )
  {
    v85.m128i_i64[0] = v34;
    v85.m128i_i64[1] = v36;
    sub_C8D5F0((__int64)&v109, (const void *)v86, (unsigned int)v110 + 1LL, 0x10u, v34, v36);
    v37 = (unsigned int)v110;
    v36 = v85.m128i_i64[1];
    v38 = v85.m128i_i64[0];
  }
  v39 = (__int64 *)&v109[v37];
  *v39 = v38;
  v39[1] = v36;
  LODWORD(v110) = v110 + 1;
  if ( v98 )
    sub_B91220((__int64)v87, v98);
  v40 = v90;
  v41 = (unsigned int)v110;
  for ( i = 1; i < *(_DWORD *)(v90 + 64); LODWORD(v110) = v110 + 1 )
  {
    v43 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v40 + 40) + 40LL * i));
    if ( v41 + 1 > (unsigned __int64)HIDWORD(v110) )
    {
      v85 = v43;
      sub_C8D5F0((__int64)&v109, (const void *)v86, v41 + 1, 0x10u, v38, v36);
      v41 = (unsigned int)v110;
      v43 = _mm_load_si128(&v85);
    }
    ++i;
    v109[v41] = v43;
    v40 = v90;
    v41 = (unsigned int)(v110 + 1);
  }
  v44 = sub_33E5110(a2, v88, v89, 1, 0);
  v98 = 0;
  v83 = v44;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v81 = v45;
  v46 = sub_2EAC4F0(*(_QWORD *)(v90 + 112));
  v47 = v90;
  v48 = (__int64)v109;
  v49 = (unsigned int)v110;
  v50 = v46;
  v51 = *(_QWORD *)(v90 + 80);
  v52 = *(__int64 **)(v90 + 112);
  v94 = v51;
  if ( v51 )
  {
    v78 = v46;
    v76 = (__int64)v109;
    v77 = (unsigned int)v110;
    v79 = v52;
    v85.m128i_i64[0] = (__int64)&v94;
    sub_B96E90((__int64)&v94, v51, 1);
    v50 = v78;
    v48 = v76;
    v49 = v77;
    v52 = v79;
  }
  else
  {
    v85.m128i_i64[0] = (__int64)&v94;
  }
  v95 = *(_DWORD *)(v47 + 72);
  v53 = sub_33EB1C0(
          a2,
          47,
          v85.m128i_i32[0],
          v83,
          v81,
          v50,
          v48,
          v49,
          v88,
          v89,
          *v52,
          v52[1],
          v52[2],
          3,
          0,
          (__int64)v87);
  v55 = v54;
  v56 = v53;
  if ( v94 )
  {
    v88 = v53;
    sub_B91220(v85.m128i_i64[0], v94);
    v56 = v88;
  }
  if ( v109 != (_OWORD *)v86 )
  {
    v88 = v56;
    _libc_free((unsigned __int64)v109);
    v56 = v88;
  }
LABEL_70:
  v88 = v56;
  sub_34161C0(a2, v90, 2, v56, 1);
  v57 = *(__int16 **)(a1 + 48);
  v58 = v88;
  v59 = *v57;
  v60 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))*((_QWORD *)v57 + 1);
  v61 = *(_QWORD *)(v88 + 48) + 16LL * v55;
  if ( v59 != *(_WORD *)v61 || *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(v61 + 8) != v60 && !v59 )
  {
    v62 = *(_QWORD *)(a1 + 80);
    v63 = v59;
    v109 = (_OWORD *)v62;
    if ( v62 )
    {
      v86 = v59;
      v87 = v60;
      sub_B96E90((__int64)&v109, v62, 1);
      v63 = v86;
      v60 = v87;
      v58 = v88;
    }
    LODWORD(v110) = *(_DWORD *)(a1 + 72);
    v64 = sub_33FAFB0(a2, v58, v55, &v109, v63, v60);
    v58 = v64;
    if ( v109 )
    {
      v88 = v64;
      sub_B91220((__int64)&v109, (__int64)v109);
      v58 = v88;
    }
  }
  v10 = v58;
LABEL_5:
  sub_302F890(v105);
  if ( (_BYTE *)v102[0] != v103 )
    _libc_free(v102[0]);
  return v10;
}
