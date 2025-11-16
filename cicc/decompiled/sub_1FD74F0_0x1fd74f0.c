// Function: sub_1FD74F0
// Address: 0x1fd74f0
//
__int64 __fastcall sub_1FD74F0(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  unsigned __int64 v4; // r14
  bool v5; // sf
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  int v10; // ebx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rsi
  unsigned __int64 v14; // rbx
  int v15; // eax
  __int64 **v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r13
  int v20; // r13d
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 **i; // rbx
  unsigned __int64 v24; // r12
  _QWORD *v25; // rdi
  __int64 v26; // rax
  char v27; // al
  unsigned __int64 v28; // r12
  _QWORD *v29; // rdi
  __int64 v30; // rax
  char v31; // cl
  unsigned __int64 v32; // r12
  _QWORD *v33; // rdi
  __int64 v34; // rax
  char v35; // al
  unsigned __int64 v36; // r12
  _QWORD *v37; // rdi
  __int64 v38; // rax
  char v39; // al
  const __m128i *v40; // rdi
  const __m128i *v41; // rsi
  const __m128i *v42; // rax
  __m128i *v43; // rax
  const __m128i *v44; // rax
  int v45; // eax
  unsigned int v46; // r12d
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // r13
  int v51; // r13d
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 *v55; // r14
  __m128i *v56; // rsi
  __int64 v57; // [rsp+10h] [rbp-370h]
  char v58; // [rsp+18h] [rbp-368h]
  __int64 v60; // [rsp+28h] [rbp-358h]
  __int64 v61; // [rsp+30h] [rbp-350h] BYREF
  __int64 v62; // [rsp+38h] [rbp-348h] BYREF
  const __m128i *v63; // [rsp+40h] [rbp-340h] BYREF
  __m128i *v64; // [rsp+48h] [rbp-338h]
  const __m128i *v65; // [rsp+50h] [rbp-330h]
  __m128i v66; // [rsp+60h] [rbp-320h] BYREF
  __m128i v67; // [rsp+70h] [rbp-310h] BYREF
  __int64 v68; // [rsp+80h] [rbp-300h]
  __int64 v69; // [rsp+90h] [rbp-2F0h] BYREF
  unsigned __int64 v70; // [rsp+98h] [rbp-2E8h]
  __int64 v71; // [rsp+A0h] [rbp-2E0h]
  __int64 v72; // [rsp+A8h] [rbp-2D8h]
  __int64 v73; // [rsp+B0h] [rbp-2D0h]
  const __m128i *v74; // [rsp+B8h] [rbp-2C8h]
  __m128i *v75; // [rsp+C0h] [rbp-2C0h]
  const __m128i *v76; // [rsp+C8h] [rbp-2B8h]
  __int64 *v77; // [rsp+D0h] [rbp-2B0h]
  __int64 v78; // [rsp+D8h] [rbp-2A8h]
  __int64 v79; // [rsp+E0h] [rbp-2A0h]
  _BYTE *v80; // [rsp+E8h] [rbp-298h]
  __int64 v81; // [rsp+F0h] [rbp-290h]
  _BYTE v82[128]; // [rsp+F8h] [rbp-288h] BYREF
  _BYTE *v83; // [rsp+178h] [rbp-208h]
  __int64 v84; // [rsp+180h] [rbp-200h]
  _BYTE v85[128]; // [rsp+188h] [rbp-1F8h] BYREF
  _BYTE *v86; // [rsp+208h] [rbp-178h]
  __int64 v87; // [rsp+210h] [rbp-170h]
  _BYTE v88[64]; // [rsp+218h] [rbp-168h] BYREF
  _BYTE *v89; // [rsp+258h] [rbp-128h]
  __int64 v90; // [rsp+260h] [rbp-120h]
  _BYTE v91[192]; // [rsp+268h] [rbp-118h] BYREF
  _BYTE *v92; // [rsp+328h] [rbp-58h]
  __int64 v93; // [rsp+330h] [rbp-50h]
  _BYTE v94[72]; // [rsp+338h] [rbp-48h] BYREF

  v2 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v4 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v61 = a2 | 4;
  v5 = *(char *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 23) < 0;
  v60 = *(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 64);
  v6 = *(_QWORD *)(a2 & 0xFFFFFFFFFFFFFFF8LL);
  v63 = 0;
  v57 = v6;
  v64 = 0;
  v65 = 0;
  v66 = 0u;
  v67 = 0u;
  LODWORD(v68) = 0;
  if ( !v5 )
  {
    v13 = -24;
    goto LABEL_8;
  }
  v7 = sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL);
  v9 = v7 + v8;
  if ( *(char *)(v2 + 23) >= 0 )
  {
    if ( (unsigned int)(v9 >> 4) )
LABEL_96:
      BUG();
LABEL_87:
    v13 = -24;
    v2 = v61 & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_8;
  }
  if ( !(unsigned int)((v9 - sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL)) >> 4) )
    goto LABEL_87;
  if ( *(char *)(v2 + 23) >= 0 )
    goto LABEL_96;
  v10 = *(_DWORD *)(sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL) + 8);
  if ( *(char *)(v2 + 23) >= 0 )
    BUG();
  v11 = sub_1648A40(a2 & 0xFFFFFFFFFFFFFFF8LL);
  v2 = v61 & 0xFFFFFFFFFFFFFFF8LL;
  v13 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v11 + v12 - 4) - v10);
LABEL_8:
  sub_1FD3FA0(
    &v63,
    -1431655765 * (unsigned int)((__int64)(v4 + v13 - (v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF))) >> 3));
  v14 = v61 & 0xFFFFFFFFFFFFFFF8LL;
  v15 = *(_DWORD *)((v61 & 0xFFFFFFFFFFFFFFF8LL) + 20);
  if ( (v61 & 4) == 0 )
  {
    v16 = (__int64 **)(v14 - 24LL * (v15 & 0xFFFFFFF));
    if ( *(char *)(v14 + 23) < 0 )
    {
      v48 = sub_1648A40(v61 & 0xFFFFFFFFFFFFFFF8LL);
      v50 = v48 + v49;
      if ( *(char *)(v14 + 23) >= 0 )
      {
        if ( (unsigned int)(v50 >> 4) )
          goto LABEL_93;
      }
      else if ( (unsigned int)((v50 - sub_1648A40(v14)) >> 4) )
      {
        if ( *(char *)(v14 + 23) < 0 )
        {
          v51 = *(_DWORD *)(sub_1648A40(v14) + 8);
          if ( *(char *)(v14 + 23) >= 0 )
            BUG();
          v52 = sub_1648A40(v14);
          v54 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v52 + v53 - 4) - v51);
LABEL_58:
          i = (__int64 **)(v54 + v14);
          if ( i == v16 )
            goto LABEL_15;
          goto LABEL_59;
        }
LABEL_93:
        BUG();
      }
    }
    v54 = -72;
    goto LABEL_58;
  }
  v16 = (__int64 **)(v14 - 24LL * (v15 & 0xFFFFFFF));
  if ( *(char *)(v14 + 23) >= 0 )
    goto LABEL_81;
  v17 = sub_1648A40(v61 & 0xFFFFFFFFFFFFFFF8LL);
  v19 = v17 + v18;
  if ( *(char *)(v14 + 23) >= 0 )
  {
    if ( (unsigned int)(v19 >> 4) )
LABEL_91:
      BUG();
    goto LABEL_81;
  }
  if ( !(unsigned int)((v19 - sub_1648A40(v14)) >> 4) )
  {
LABEL_81:
    v54 = -24;
    goto LABEL_58;
  }
  if ( *(char *)(v14 + 23) >= 0 )
    goto LABEL_91;
  v20 = *(_DWORD *)(sub_1648A40(v14) + 8);
  if ( *(char *)(v14 + 23) >= 0 )
    BUG();
  v21 = sub_1648A40(v14);
  for ( i = (__int64 **)(-24 - 24LL * (unsigned int)(*(_DWORD *)(v21 + v22 - 4) - v20) + v14); v16 != i; v16 += 3 )
  {
LABEL_59:
    v55 = *v16;
    if ( !(unsigned __int8)sub_1642FB0(**v16) )
    {
      v66.m128i_i64[0] = (__int64)v55;
      v67.m128i_i64[1] = *v55;
      sub_20A1C00(
        &v66,
        &v61,
        0xAAAAAAAAAAAAAAABLL
      * ((__int64)((__int64)&v16[3 * (*(_DWORD *)((v61 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF)]
                 - (v61 & 0xFFFFFFFFFFFFFFF8LL)) >> 3));
      v56 = v64;
      if ( v64 == v65 )
      {
        sub_1D27190(&v63, v64, &v66);
      }
      else
      {
        if ( v64 )
        {
          *v64 = _mm_loadu_si128(&v66);
          v56[1] = _mm_loadu_si128(&v67);
          v56[2].m128i_i64[0] = v68;
          v56 = v64;
        }
        v64 = (__m128i *)((char *)v56 + 40);
      }
    }
  }
LABEL_15:
  if ( (*(_WORD *)(a2 + 18) & 3u) - 1 <= 1 )
    v58 = sub_20C8B80(v61, a1[11]);
  else
    v58 = 0;
  v70 = 0xFFFFFFFF00000020LL;
  v81 = 0x1000000000LL;
  v84 = 0x1000000000LL;
  v87 = 0x1000000000LL;
  v90 = 0x400000000LL;
  v93 = 0x400000000LL;
  v72 = *(_QWORD *)(a2 - 24);
  v83 = v85;
  v86 = v88;
  v24 = v61 & 0xFFFFFFFFFFFFFFF8LL;
  v71 = 0;
  v25 = (_QWORD *)((v61 & 0xFFFFFFFFFFFFFFF8LL) + 56);
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v80 = v82;
  v89 = v91;
  v92 = v94;
  v69 = v57;
  if ( (v61 & 4) != 0 )
  {
    if ( !(unsigned __int8)sub_1560260(v25, 0, 12) )
    {
      v26 = *(_QWORD *)(v24 - 24);
      if ( !*(_BYTE *)(v26 + 16) )
      {
LABEL_20:
        v62 = *(_QWORD *)(v26 + 112);
        v27 = sub_1560260(&v62, 0, 12);
        goto LABEL_21;
      }
      goto LABEL_79;
    }
  }
  else if ( !(unsigned __int8)sub_1560260(v25, 0, 12) )
  {
    v26 = *(_QWORD *)(v24 - 72);
    if ( !*(_BYTE *)(v26 + 16) )
      goto LABEL_20;
LABEL_79:
    v27 = 0;
    goto LABEL_21;
  }
  v27 = 1;
LABEL_21:
  LOBYTE(v70) = v70 & 0xF7 | (8 * (v27 & 1));
  v28 = v61 & 0xFFFFFFFFFFFFFFF8LL;
  v29 = (_QWORD *)((v61 & 0xFFFFFFFFFFFFFFF8LL) + 56);
  if ( (v61 & 4) != 0 )
  {
    if ( !(unsigned __int8)sub_1560260(v29, -1, 29) )
    {
      v30 = *(_QWORD *)(v28 - 24);
      if ( !*(_BYTE *)(v30 + 16) )
      {
LABEL_24:
        v62 = *(_QWORD *)(v30 + 112);
        v31 = sub_1560260(&v62, -1, 29);
        goto LABEL_25;
      }
      goto LABEL_76;
    }
LABEL_67:
    v31 = 1;
    goto LABEL_25;
  }
  if ( (unsigned __int8)sub_1560260(v29, -1, 29) )
    goto LABEL_67;
  v30 = *(_QWORD *)(v28 - 72);
  if ( !*(_BYTE *)(v30 + 16) )
    goto LABEL_24;
LABEL_76:
  v31 = 0;
LABEL_25:
  v32 = v61 & 0xFFFFFFFFFFFFFFF8LL;
  v33 = (_QWORD *)((v61 & 0xFFFFFFFFFFFFFFF8LL) + 56);
  LOBYTE(v70) = v70 & 0xEB | ((16 * v31) | (4 * (*(_DWORD *)(v60 + 8) >> 8 != 0))) & 0x14;
  LOBYTE(v70) = (32 * (*(_QWORD *)((v61 & 0xFFFFFFFFFFFFFFF8LL) + 8) != 0)) | v70 & 0xDF;
  if ( (v61 & 4) != 0 )
  {
    if ( !(unsigned __int8)sub_1560260(v33, 0, 40) )
    {
      v34 = *(_QWORD *)(v32 - 24);
      if ( !*(_BYTE *)(v34 + 16) )
      {
LABEL_28:
        v62 = *(_QWORD *)(v34 + 112);
        v35 = sub_1560260(&v62, 0, 40);
        goto LABEL_29;
      }
      goto LABEL_70;
    }
  }
  else if ( !(unsigned __int8)sub_1560260(v33, 0, 40) )
  {
    v34 = *(_QWORD *)(v32 - 72);
    if ( !*(_BYTE *)(v34 + 16) )
      goto LABEL_28;
LABEL_70:
    v35 = 0;
    goto LABEL_29;
  }
  v35 = 1;
LABEL_29:
  LOBYTE(v70) = v35 & 1 | v70 & 0xFE;
  v36 = v61 & 0xFFFFFFFFFFFFFFF8LL;
  v37 = (_QWORD *)((v61 & 0xFFFFFFFFFFFFFFF8LL) + 56);
  if ( (v61 & 4) != 0 )
  {
    if ( !(unsigned __int8)sub_1560260(v37, 0, 58) )
    {
      v38 = *(_QWORD *)(v36 - 24);
      if ( !*(_BYTE *)(v38 + 16) )
      {
LABEL_32:
        v62 = *(_QWORD *)(v38 + 112);
        v39 = sub_1560260(&v62, 0, 58);
        goto LABEL_33;
      }
      goto LABEL_73;
    }
  }
  else if ( !(unsigned __int8)sub_1560260(v37, 0, 58) )
  {
    v38 = *(_QWORD *)(v36 - 72);
    if ( !*(_BYTE *)(v38 + 16) )
      goto LABEL_32;
LABEL_73:
    v39 = 0;
    goto LABEL_33;
  }
  v39 = 1;
LABEL_33:
  v40 = v74;
  v41 = v76;
  LOBYTE(v70) = v70 & 0xFD | (2 * (v39 & 1));
  LODWORD(v71) = (*(unsigned __int16 *)((v61 & 0xFFFFFFFFFFFFFFF8LL) + 18) >> 2) & 0x3FFFDFFF;
  v42 = v63;
  v63 = 0;
  v74 = v42;
  v43 = v64;
  v64 = 0;
  v75 = v43;
  v44 = v65;
  v65 = 0;
  v76 = v44;
  if ( v40 )
    j_j___libc_free_0(v40, (char *)v41 - (char *)v40);
  v45 = *(_DWORD *)(v60 + 12);
  v77 = &v61;
  HIDWORD(v70) = v45 - 1;
  BYTE1(v70) = v58;
  v46 = sub_1FD6490(a1, (__int64)&v69);
  if ( v92 != v94 )
    _libc_free((unsigned __int64)v92);
  if ( v89 != v91 )
    _libc_free((unsigned __int64)v89);
  if ( v86 != v88 )
    _libc_free((unsigned __int64)v86);
  if ( v83 != v85 )
    _libc_free((unsigned __int64)v83);
  if ( v80 != v82 )
    _libc_free((unsigned __int64)v80);
  if ( v74 )
    j_j___libc_free_0(v74, (char *)v76 - (char *)v74);
  if ( v63 )
    j_j___libc_free_0(v63, (char *)v65 - (char *)v63);
  return v46;
}
