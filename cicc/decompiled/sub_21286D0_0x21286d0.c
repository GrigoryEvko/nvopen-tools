// Function: sub_21286D0
// Address: 0x21286d0
//
__int64 __fastcall sub_21286D0(
        __int64 a1,
        unsigned __int64 a2,
        __m128i a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        int a8,
        int a9)
{
  unsigned int v9; // r10d
  __int64 v12; // rax
  __int128 v13; // xmm1
  __m128i *v14; // rsi
  char *v15; // rax
  __int64 v16; // rsi
  unsigned __int8 v17; // cl
  __int64 v18; // r15
  __int64 v19; // r13
  __int64 v20; // r12
  unsigned int v21; // r13d
  __int64 v22; // r12
  __int64 *v23; // rdx
  __int64 *v24; // rax
  __int64 v25; // r8
  __int64 v26; // r15
  unsigned int v27; // esi
  unsigned __int64 v28; // r9
  __int64 v29; // rdx
  __int64 v30; // r10
  _QWORD *v31; // rax
  __int64 *v32; // rcx
  int v33; // edx
  int v34; // edi
  unsigned __int64 v35; // rdx
  __int64 v36; // rax
  const void **v37; // r15
  const __m128i *v38; // r9
  unsigned __int64 v39; // rdx
  unsigned int v40; // eax
  unsigned int v41; // r11d
  char v42; // al
  __int128 v43; // rax
  unsigned int v44; // edx
  unsigned int v45; // r11d
  unsigned int v46; // edx
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  unsigned __int64 v49; // r13
  __int64 v50; // r12
  __int64 v51; // rax
  int v52; // eax
  unsigned int v53; // r11d
  unsigned int v54; // edx
  unsigned int v56; // r10d
  char v57; // r13
  __int64 v58; // r12
  unsigned int v59; // eax
  const __m128i *v60; // rdx
  __int128 *v61; // rax
  int v62; // eax
  __int64 v63; // r15
  int v64; // r13d
  char v65; // di
  unsigned int v66; // eax
  char v67; // al
  __int128 v68; // [rsp-10h] [rbp-1F0h]
  unsigned int v69; // [rsp+8h] [rbp-1D8h]
  __m128i *v70; // [rsp+18h] [rbp-1C8h]
  __int64 v71; // [rsp+20h] [rbp-1C0h]
  __int64 v72; // [rsp+28h] [rbp-1B8h]
  unsigned int v73; // [rsp+30h] [rbp-1B0h]
  __int64 v74; // [rsp+30h] [rbp-1B0h]
  unsigned int v76; // [rsp+40h] [rbp-1A0h]
  unsigned int v77; // [rsp+40h] [rbp-1A0h]
  unsigned int v78; // [rsp+40h] [rbp-1A0h]
  unsigned int v79; // [rsp+48h] [rbp-198h]
  __int64 *v80; // [rsp+48h] [rbp-198h]
  __int64 v81; // [rsp+48h] [rbp-198h]
  unsigned int v82; // [rsp+50h] [rbp-190h]
  unsigned int v83; // [rsp+60h] [rbp-180h]
  unsigned int v84; // [rsp+68h] [rbp-178h]
  char v85; // [rsp+68h] [rbp-178h]
  __int64 v86; // [rsp+68h] [rbp-178h]
  __int64 v87; // [rsp+68h] [rbp-178h]
  __int64 v88; // [rsp+68h] [rbp-178h]
  unsigned __int8 v89; // [rsp+70h] [rbp-170h]
  unsigned __int64 v90; // [rsp+70h] [rbp-170h]
  __int64 v91; // [rsp+70h] [rbp-170h]
  unsigned int v92; // [rsp+70h] [rbp-170h]
  unsigned __int64 v93; // [rsp+78h] [rbp-168h]
  __int64 *v94; // [rsp+80h] [rbp-160h]
  __int64 *v95; // [rsp+90h] [rbp-150h]
  char v96; // [rsp+CBh] [rbp-115h] BYREF
  unsigned int v97; // [rsp+CCh] [rbp-114h] BYREF
  __int64 v98; // [rsp+D0h] [rbp-110h] BYREF
  int v99; // [rsp+D8h] [rbp-108h]
  __int64 v100; // [rsp+E0h] [rbp-100h] BYREF
  __int64 v101; // [rsp+E8h] [rbp-F8h]
  __int64 v102; // [rsp+F0h] [rbp-F0h] BYREF
  __int64 v103; // [rsp+F8h] [rbp-E8h]
  __int64 v104; // [rsp+100h] [rbp-E0h] BYREF
  __int64 v105; // [rsp+108h] [rbp-D8h]
  const void **v106; // [rsp+110h] [rbp-D0h]
  __int128 *v107; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v108; // [rsp+128h] [rbp-B8h]
  __int64 v109[22]; // [rsp+130h] [rbp-B0h] BYREF

  v12 = *(_QWORD *)(a2 + 32);
  v13 = (__int128)_mm_loadu_si128((const __m128i *)(v12 + 40));
  v71 = *(_QWORD *)v12;
  v14 = *(__m128i **)(v12 + 8);
  v15 = *(char **)(a2 + 40);
  v70 = v14;
  v16 = *(_QWORD *)(a2 + 72);
  v17 = *v15;
  v18 = *((_QWORD *)v15 + 1);
  v98 = v16;
  if ( v16 )
  {
    v84 = v9;
    v89 = v17;
    sub_1623A60((__int64)&v98, v16, 2);
    v9 = v84;
    v17 = v89;
  }
  v19 = *(_QWORD *)a1;
  v99 = *(_DWORD *)(a2 + 64);
  v20 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL);
  LOBYTE(v100) = v17;
  v101 = v18;
  if ( v17 )
  {
    v85 = *(_BYTE *)(v19 + v17 + 1155);
    v21 = *(unsigned __int8 *)(v19 + v17 + 1040);
  }
  else
  {
    v92 = v9;
    if ( sub_1F58D20((__int64)&v100) )
    {
      LOBYTE(v107) = 0;
      v108 = 0;
      LOBYTE(v102) = 0;
      sub_1F426C0(v19, v20, (unsigned int)v100, v18, (__int64)&v107, (unsigned int *)&v104, &v102);
      v56 = v92;
      v85 = v102;
    }
    else
    {
      sub_1F40D10((__int64)&v107, v19, v20, v100, v101);
      v56 = v92;
      LOBYTE(v102) = v108;
      v103 = v109[0];
      if ( (_BYTE)v108 )
      {
        v85 = *(_BYTE *)(v19 + (unsigned __int8)v108 + 1155);
      }
      else
      {
        v87 = v109[0];
        if ( sub_1F58D20((__int64)&v102) )
        {
          LOBYTE(v107) = 0;
          v108 = 0;
          LOBYTE(v97) = 0;
          sub_1F426C0(v19, v20, (unsigned int)v102, v87, (__int64)&v107, (unsigned int *)&v104, &v97);
          v56 = v92;
          v85 = v97;
        }
        else
        {
          sub_1F40D10((__int64)&v107, v19, v20, v102, v103);
          v56 = v92;
          LOBYTE(v104) = v108;
          v105 = v109[0];
          if ( (_BYTE)v108 )
          {
            v57 = *(_BYTE *)(v19 + (unsigned __int8)v108 + 1155);
          }
          else
          {
            v88 = v109[0];
            if ( sub_1F58D20((__int64)&v104) )
            {
              LOBYTE(v107) = 0;
              v108 = 0;
              v96 = 0;
              sub_1F426C0(v19, v20, (unsigned int)v104, v88, (__int64)&v107, &v97, &v96);
              v57 = v96;
              v56 = v92;
            }
            else
            {
              sub_1F40D10((__int64)&v107, v19, v20, v104, v105);
              v67 = sub_1D5E9F0(v19, v20, (unsigned __int8)v108, v109[0]);
              v56 = v92;
              v57 = v67;
            }
          }
          v85 = v57;
        }
      }
    }
    v77 = v56;
    v74 = *(_QWORD *)a1;
    v58 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL);
    LOBYTE(v102) = 0;
    v103 = v18;
    if ( sub_1F58D20((__int64)&v102) )
    {
      LOBYTE(v107) = 0;
      LOBYTE(v100) = 0;
      v108 = 0;
      v59 = sub_1F426C0(v74, v58, (unsigned int)v102, v18, (__int64)&v107, (unsigned int *)&v104, &v100);
      v9 = v77;
      v21 = v59;
    }
    else
    {
      v62 = sub_1F58D40((__int64)&v102);
      v63 = v103;
      v64 = v62;
      v104 = v102;
      v81 = v102;
      v105 = v103;
      if ( sub_1F58D20((__int64)&v104) )
      {
        LOBYTE(v107) = 0;
        v108 = 0;
        LOBYTE(v97) = 0;
        sub_1F426C0(v74, v58, (unsigned int)v104, v63, (__int64)&v107, (unsigned int *)&v100, &v97);
        v65 = v97;
      }
      else
      {
        sub_1F40D10((__int64)&v107, v74, v58, v81, v63);
        v65 = sub_1D5E9F0(v74, v58, (unsigned __int8)v108, v109[0]);
      }
      v66 = sub_2127930(v65);
      v21 = (v66 + v64 - 1) / v66;
    }
  }
  v22 = 16LL * v21;
  v107 = (__int128 *)v109;
  v108 = 0x800000000LL;
  if ( v21 <= 8uLL )
  {
    LODWORD(v108) = v21;
    v23 = &v109[(unsigned __int64)v22 / 8];
    v24 = v109;
    if ( &v109[(unsigned __int64)v22 / 8] == v109 )
      goto LABEL_10;
    goto LABEL_7;
  }
  v78 = v9;
  sub_16CD150((__int64)&v107, v109, v21, 16, a8, a9);
  v24 = (__int64 *)v107;
  LODWORD(v108) = v21;
  v9 = v78;
  v23 = (__int64 *)&v107[(unsigned __int64)v22 / 0x10];
  if ( &v107[(unsigned __int64)v22 / 0x10] != v107 )
  {
    do
    {
LABEL_7:
      if ( v24 )
      {
        *v24 = 0;
        *((_DWORD *)v24 + 2) = 0;
      }
      v24 += 2;
    }
    while ( v24 != v23 );
LABEL_10:
    if ( !v21 )
      goto LABEL_16;
  }
  v25 = v71;
  v26 = 0;
  v27 = v9;
  v28 = (unsigned __int64)v70;
  do
  {
    v29 = *(_QWORD *)(a2 + 32);
    v30 = *(_QWORD *)(*(_QWORD *)(v29 + 120) + 88LL);
    v31 = *(_QWORD **)(v30 + 24);
    if ( *(_DWORD *)(v30 + 32) > 0x40u )
      v31 = (_QWORD *)*v31;
    LOBYTE(v27) = v85;
    v90 = v28;
    v32 = sub_1D38F20(
            *(__int64 **)(a1 + 8),
            v27,
            0,
            (__int64)&v98,
            v25,
            v28,
            *(double *)a3.m128i_i64,
            *(double *)&v13,
            a5,
            v13,
            *(_OWORD *)(v29 + 80),
            (unsigned int)v31);
    v34 = v33;
    v35 = (unsigned __int64)v107;
    v36 = v26++;
    v28 = v90 & 0xFFFFFFFF00000000LL | 1;
    *(_QWORD *)&v107[v36] = v32;
    *(_DWORD *)(v35 + v36 * 16 + 8) = v34;
    v25 = *(_QWORD *)&v107[v36];
  }
  while ( v21 > (unsigned int)v26 );
  v71 = *(_QWORD *)&v107[v36];
  v70 = (__m128i *)(v90 & 0xFFFFFFFF00000000LL | 1);
LABEL_16:
  if ( *(_BYTE *)sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL)) )
  {
    v60 = (const __m128i *)v107;
    v61 = &v107[(unsigned int)v108];
    if ( v61 != v107 )
    {
      while ( v60 < (const __m128i *)--v61 )
      {
        a3 = _mm_loadu_si128(v60++);
        v60[-1].m128i_i64[0] = *(_QWORD *)v61;
        v60[-1].m128i_i32[2] = *((_DWORD *)v61 + 2);
        *(_QWORD *)v61 = a3.m128i_i64[0];
        *((_DWORD *)v61 + 2) = a3.m128i_i32[2];
      }
    }
  }
  sub_1F40D10(
    (__int64)&v104,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v37 = v106;
  v82 = (unsigned __int8)v105;
  v91 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          143,
          (__int64)&v98,
          (unsigned __int8)v105,
          v106,
          0,
          *(double *)a3.m128i_i64,
          *(double *)&v13,
          *(double *)a5.m128i_i64,
          *v107);
  v93 = v39;
  if ( v21 > 1 )
  {
    v40 = sub_2127930(v85);
    v86 = 16;
    v73 = v40;
    v41 = v69;
    v83 = v40;
    v72 = 16LL * v21;
    do
    {
      v76 = v41;
      v47 = sub_1D309E0(
              *(__int64 **)(a1 + 8),
              143,
              (__int64)&v98,
              v82,
              v37,
              0,
              *(double *)a3.m128i_i64,
              *(double *)&v13,
              *(double *)a5.m128i_i64,
              v107[(unsigned __int64)v86 / 0x10]);
      v49 = v48;
      v50 = v47;
      v80 = *(__int64 **)(a1 + 8);
      v51 = sub_1E0A0C0(v80[4]);
      v52 = sub_15A9520(v51, 0);
      v53 = v76;
      v54 = 8 * v52;
      if ( 8 * v52 == 32 )
      {
        v42 = 5;
      }
      else if ( v54 <= 0x20 )
      {
        v42 = 3;
        if ( v54 != 8 )
          v42 = 4 * (v54 == 16);
      }
      else
      {
        v42 = 6;
        if ( v54 != 64 )
        {
          v42 = 0;
          if ( v54 == 128 )
            v42 = 7;
        }
      }
      LOBYTE(v53) = v42;
      *(_QWORD *)&v43 = sub_1D38BB0((__int64)v80, v83, (__int64)&v98, v53, 0, 0, a3, *(double *)&v13, a5, 0);
      v95 = sub_1D332F0(
              v80,
              122,
              (__int64)&v98,
              v82,
              v37,
              0,
              *(double *)a3.m128i_i64,
              *(double *)&v13,
              a5,
              v50,
              v49,
              v43);
      v79 = v45;
      *((_QWORD *)&v68 + 1) = v44 | v49 & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v68 = v95;
      v94 = sub_1D332F0(
              *(__int64 **)(a1 + 8),
              119,
              (__int64)&v98,
              v82,
              v37,
              0,
              *(double *)a3.m128i_i64,
              *(double *)&v13,
              a5,
              v91,
              v93,
              v68);
      v41 = v79;
      v91 = (__int64)v94;
      v86 += 16;
      v83 += v73;
      v93 = v46 | v93 & 0xFFFFFFFF00000000LL;
    }
    while ( v86 != v72 );
  }
  sub_2013400(a1, a2, 1, v71, v70, v38);
  if ( v107 != (__int128 *)v109 )
    _libc_free((unsigned __int64)v107);
  if ( v98 )
    sub_161E7C0((__int64)&v98, v98);
  return v91;
}
