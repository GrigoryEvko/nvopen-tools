// Function: sub_92BD50
// Address: 0x92bd50
//
__int64 __fastcall sub_92BD50(__int64 *a1, __int64 a2, char a3, __int64 a4, unsigned __int8 a5, char a6, _DWORD *a7)
{
  __int64 v7; // r10
  __int64 v9; // rdi
  __int64 v11; // r15
  __int64 v12; // r13
  unsigned __int8 v14; // dl
  unsigned __int8 v15; // al
  __int64 v16; // rdi
  bool v17; // cf
  char v18; // al
  unsigned int **v20; // r14
  unsigned int *v21; // rdi
  __int64 (__fastcall *v22)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v23; // rax
  __int64 v24; // r10
  unsigned int *v25; // rbx
  __int64 v26; // r12
  __int64 v27; // rdx
  __int64 v28; // rsi
  unsigned int v29; // eax
  __int64 v30; // rax
  unsigned int **v31; // r10
  __int64 v32; // rdi
  __int64 v33; // r15
  unsigned int v34; // esi
  __int64 v35; // rax
  unsigned int **v36; // rdi
  __int64 v37; // rdi
  unsigned __int8 v38; // al
  __int64 v39; // rax
  __int64 v40; // rdi
  _BYTE *v41; // rax
  int v42; // eax
  char v43; // al
  _BYTE *v44; // rax
  unsigned int **v45; // rdi
  unsigned int **v46; // rdi
  __int64 v47; // rdi
  char v48; // al
  int v49; // edx
  const char *v50; // rdi
  __int64 v51; // rcx
  unsigned int v52; // eax
  __int64 v53; // rsi
  char *v54; // r9
  __int64 v55; // rdi
  _BYTE *v56; // rcx
  unsigned int v57; // eax
  __int64 v58; // rsi
  __int64 v59; // rax
  char *v60; // rax
  __int64 v61; // r15
  __int64 v62; // rax
  char *v63; // rdi
  __int64 v64; // rdi
  char v65; // al
  unsigned int **v66; // rdi
  __int64 v67; // rdi
  unsigned int **v68; // r14
  unsigned int v69; // r15d
  unsigned int v70; // esi
  __int64 v71; // rbx
  __int64 v72; // rdx
  __int64 v73; // rcx
  __m128i *v74; // rax
  __m128i si128; // xmm0
  __int64 v76; // rdi
  __int64 v77; // rax
  char *v78; // rax
  unsigned int **v79; // [rsp+0h] [rbp-B0h]
  __int64 v81; // [rsp+8h] [rbp-A8h]
  __int64 v82; // [rsp+8h] [rbp-A8h]
  __int64 v83; // [rsp+8h] [rbp-A8h]
  unsigned int v84; // [rsp+8h] [rbp-A8h]
  _BYTE *v85; // [rsp+8h] [rbp-A8h]
  __int64 v86; // [rsp+8h] [rbp-A8h]
  __int64 v87; // [rsp+8h] [rbp-A8h]
  unsigned int v88; // [rsp+18h] [rbp-98h]
  unsigned int v89[8]; // [rsp+20h] [rbp-90h] BYREF
  char v90; // [rsp+40h] [rbp-70h]
  char v91; // [rsp+41h] [rbp-6Fh]
  char *v92; // [rsp+50h] [rbp-60h] BYREF
  __int64 v93; // [rsp+58h] [rbp-58h]
  unsigned __int64 v94; // [rsp+60h] [rbp-50h]
  _BYTE v95[8]; // [rsp+68h] [rbp-48h] BYREF
  __int16 v96; // [rsp+70h] [rbp-40h]

  v7 = 0;
  if ( a6 )
    return v7;
  v9 = a1[2];
  v11 = *(_QWORD *)(a2 + 8);
  v12 = a2;
  if ( a4 == sub_BCB2A0(v9) )
  {
    v37 = *(_QWORD *)(a2 + 8);
    v38 = *(_BYTE *)(v37 + 8);
    if ( v38 <= 3u || v38 == 5 || (v43 = v38 & 0xFD, v43 == 4) )
    {
      v39 = sub_AD6530(v37);
      v40 = a1[1];
      v89[1] = 0;
      v92 = "tobool";
      v96 = 259;
      v41 = (_BYTE *)sub_B35C90(v40, 14, a2, v39, &v92, 0, v89[0], 0);
      v7 = (__int64)v41;
      if ( unk_4D04700 && *v41 > 0x1Cu )
      {
        v85 = v41;
        v42 = sub_B45210(v41);
        sub_B45150(v85, v42 | 1u);
        return (__int64)v85;
      }
    }
    else
    {
      if ( v43 != 12 )
        sub_91B8A0("unexpected type when converting to boolean!", a7, 1);
      if ( *(_BYTE *)a2 == 68
        && (v71 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL), v71 == sub_BCB2A0(*(_QWORD *)(*a1 + 40))) )
      {
        v7 = *(_QWORD *)(a2 - 32);
        if ( !*(_QWORD *)(a2 + 16) )
        {
          v87 = *(_QWORD *)(a2 - 32);
          sub_B43D60(a2, a2, v72, v73);
          return v87;
        }
      }
      else
      {
        v44 = (_BYTE *)sub_AD6530(*(_QWORD *)(a2 + 8));
        v45 = (unsigned int **)a1[1];
        v92 = "tobool";
        v96 = 259;
        return sub_92B530(v45, 0x21u, a2, v44, (__int64)&v92);
      }
    }
    return v7;
  }
  if ( v11 == a4 )
    return a2;
  v14 = *(_BYTE *)(a4 + 8);
  v15 = *(_BYTE *)(v11 + 8);
  if ( v14 == 14 )
  {
    if ( v15 == 14 )
    {
      v92 = "conv";
      v66 = (unsigned int **)a1[1];
      v96 = 259;
      return sub_929600(v66, 0x31u, a2, a4, (__int64)&v92, 0, v89[0], 0);
    }
    else
    {
      if ( v15 != 12 )
        sub_91B8A0("unexpected destination type for cast from pointer type", a7, 1);
      v29 = sub_91B6E0(v9);
      v30 = sub_BCCE00(a1[2], v29);
      v31 = (unsigned int **)a1[1];
      v32 = *(_QWORD *)(a2 + 8);
      v33 = v30;
      v92 = "conv";
      v79 = v31;
      v96 = 259;
      v84 = sub_BCB060(v32);
      v34 = 39 - ((a3 == 0) - 1);
      if ( v84 > (unsigned int)sub_BCB060(v33) )
        v34 = 38;
      v35 = sub_929600(v79, v34, v12, v33, (__int64)&v92, 0, v89[0], 0);
      v36 = (unsigned int **)a1[1];
      v92 = "conv";
      v96 = 259;
      return sub_929600(v36, 0x30u, v35, a4, (__int64)&v92, 0, v89[0], 0);
    }
  }
  if ( v15 == 14 )
  {
    if ( v14 != 12 )
      sub_91B8A0("unexpected non-integer type for cast from pointer type!", a7, 1);
    v92 = "conv";
    v46 = (unsigned int **)a1[1];
    v96 = 259;
    return sub_929600(v46, 0x2Fu, a2, a4, (__int64)&v92, 0, v89[0], 0);
  }
  if ( v15 == 12 )
  {
    if ( v14 == 12 )
    {
      v67 = *(_QWORD *)(a2 + 8);
      v68 = (unsigned int **)a1[1];
      v92 = "conv";
      v96 = 259;
      v69 = sub_BCB060(v67);
      v70 = 39 - ((a3 == 0) - 1);
      if ( v69 > (unsigned int)sub_BCB060(a4) )
        v70 = 38;
      return sub_929600(v68, v70, v12, a4, (__int64)&v92, 0, v89[0], 0);
    }
    if ( a3 )
    {
      v47 = a1[1];
      v92 = "conv";
      v96 = 259;
      if ( !*(_BYTE *)(v47 + 108) )
        return sub_929600((unsigned int **)v47, 0x2Cu, a2, a4, (__int64)&v92, 0, v89[0], 0);
      LOBYTE(v89[1]) = 0;
      return sub_B358C0(v47, 136, a2, a4, v89[0], (__int64)&v92, 0, 0);
    }
    if ( !unk_4D04630 && v14 == 2 && *(_DWORD *)(v11 + 8) >> 8 == 64 )
    {
      v92 = v95;
      v93 = 0;
      v94 = 16;
      sub_C8D290(&v92, v95, 17, 1);
      v74 = (__m128i *)&v92[v93];
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F10FE0);
      v74[1].m128i_i8[0] = 110;
      *v74 = si128;
      v76 = *a1;
      v93 += 17;
      v77 = sub_92BCC0(v76, a2, a4, &v92);
      v63 = v92;
      v7 = v77;
      if ( v92 == v95 )
        return v7;
      goto LABEL_75;
    }
    v20 = (unsigned int **)a1[1];
    v91 = 1;
    *(_QWORD *)v89 = "conv";
    v90 = 3;
    if ( *((_BYTE *)v20 + 108) )
    {
      BYTE4(v92) = 0;
      return sub_B358C0((_DWORD)v20, 141, a2, a4, (_DWORD)v92, (__int64)v89, 0, 0);
    }
    v21 = v20[10];
    v22 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v21 + 120LL);
    if ( v22 == sub_920130 )
    {
      if ( *(_BYTE *)a2 > 0x15u )
      {
LABEL_27:
        v96 = 257;
        v23 = sub_BD2C40(72, unk_3F10A14);
        v24 = v23;
        if ( v23 )
        {
          v81 = v23;
          sub_B51830(v23, a2, a4, &v92, 0, 0);
          v24 = v81;
        }
        v82 = v24;
        (*(void (__fastcall **)(unsigned int *, __int64, unsigned int *, unsigned int *, unsigned int *))(*(_QWORD *)v20[11] + 16LL))(
          v20[11],
          v24,
          v89,
          v20[7],
          v20[8]);
        v25 = *v20;
        v7 = v82;
        v26 = (__int64)&(*v20)[4 * *((unsigned int *)v20 + 2)];
        if ( *v20 != (unsigned int *)v26 )
        {
          do
          {
            v27 = *((_QWORD *)v25 + 1);
            v28 = *v25;
            v25 += 4;
            v83 = v7;
            sub_B99FD0(v7, v28, v27);
            v7 = v83;
          }
          while ( (unsigned int *)v26 != v25 );
        }
        return v7;
      }
      if ( (unsigned __int8)sub_AC4810(43) )
        v7 = sub_ADAB70(43, a2, a4, 0);
      else
        v7 = sub_AA93C0(43, a2, a4);
    }
    else
    {
      v7 = v22((__int64)v21, 43u, (_BYTE *)a2, a4);
    }
    if ( v7 )
      return v7;
    goto LABEL_27;
  }
  if ( v15 > 3u && v15 != 5 && (v15 & 0xFD) != 4 )
    sub_91B8A0("expected floating point source type in cast!", a7, 1);
  if ( v14 != 12 )
  {
    if ( v14 > 3u && v14 != 5 && (v14 & 0xFD) != 4 )
      sub_91B8A0("expected floating point destination type in cast!", a7, 1);
    v16 = a1[1];
    v17 = v14 < *(_BYTE *)(v11 + 8);
    v18 = *(_BYTE *)(v16 + 108);
    v92 = "conv";
    v96 = 259;
    *(_QWORD *)v89 = v88;
    if ( v17 )
    {
      if ( v18 )
        return sub_B358C0(v16, 113, a2, a4, v88, (__int64)&v92, 0, 0);
      else
        return sub_929600((unsigned int **)v16, 0x2Du, a2, a4, (__int64)&v92, 0, v88, 0);
    }
    else if ( v18 )
    {
      return sub_B358C0(v16, 110, a2, a4, v88, (__int64)&v92, 0, 0);
    }
    else
    {
      return sub_929600((unsigned int **)v16, 0x2Eu, a2, a4, (__int64)&v92, 0, v88, 0);
    }
  }
  if ( unk_4D04630 || (v48 = *(_BYTE *)(v11 + 8), v49 = *(_DWORD *)(a4 + 8) >> 8, v49 == 128) || v48 == 5 )
  {
    v64 = a1[1];
    v65 = *(_BYTE *)(v64 + 108);
    v92 = "conv";
    v96 = 259;
    if ( a5 )
    {
      if ( v65 )
      {
        LOBYTE(v89[1]) = 0;
        return sub_B358C0(v64, 111, a2, a4, v89[0], (__int64)&v92, 0, 0);
      }
      else
      {
        return sub_929600((unsigned int **)v64, 0x2Au, a2, a4, (__int64)&v92, 0, v89[0], 0);
      }
    }
    else if ( v65 )
    {
      LOBYTE(v89[1]) = 0;
      return sub_B358C0(v64, 112, a2, a4, v89[0], (__int64)&v92, 0, 0);
    }
    else
    {
      return sub_929600((unsigned int **)v64, 0x29u, a2, a4, (__int64)&v92, 0, v89[0], 0);
    }
  }
  v50 = "__nv_double";
  v94 = 16;
  v51 = (v48 == 3) + 10LL;
  v92 = v95;
  if ( v48 != 3 )
    v50 = "__nv_float";
  *(_QWORD *)&v95[v51 - 8] = *(_QWORD *)&v50[v51 - 8];
  v52 = 0;
  do
  {
    v53 = v52;
    v52 += 8;
    *(_QWORD *)&v95[v53] = *(_QWORD *)&v50[v53];
  }
  while ( v52 < (((_DWORD)v51 - 1) & 0xFFFFFFF8) );
  v54 = (char *)"2";
  v93 = v51;
  v55 = (a5 == 0) + 1LL;
  if ( !a5 )
    v54 = "2u";
  v56 = &v95[v51];
  v57 = 0;
  if ( (a5 == 0) != -1 )
  {
    do
    {
      v58 = v57++;
      v56[v58] = v54[v58];
    }
    while ( v57 < (unsigned int)v55 );
  }
  v59 = v55 + v93;
  v93 += v55;
  if ( v49 == 64 )
  {
    if ( v94 < v59 + 5 )
    {
      sub_C8D290(&v92, v95, v59 + 5, 1);
      v78 = &v92[v93];
      *(_DWORD *)&v92[v93] = 1918856300;
    }
    else
    {
      v78 = &v92[v59];
      *(_DWORD *)v78 = 1918856300;
    }
    v78[4] = 122;
    v93 += 5;
    a2 = v12;
    v7 = sub_92BCC0(*a1, v12, a4, &v92);
  }
  else
  {
    if ( v94 < v59 + 6 )
    {
      sub_C8D290(&v92, v95, v59 + 6, 1);
      v60 = &v92[v93];
      *(_DWORD *)&v92[v93] = 1601465961;
    }
    else
    {
      v60 = &v92[v59];
      *(_DWORD *)v60 = 1601465961;
    }
    *((_WORD *)v60 + 2) = 31346;
    v93 += 6;
    a2 = v12;
    v61 = sub_BCB2D0(a1[2]);
    v62 = sub_92BCC0(*a1, v12, v61, &v92);
    v7 = v62;
    if ( v61 != a4 )
    {
      a2 = v62;
      v7 = sub_92BD50((_DWORD)a1, v62, a5, a4, a5, 0, (__int64)a7);
    }
  }
  v63 = v92;
  if ( v92 != v95 )
  {
LABEL_75:
    v86 = v7;
    _libc_free(v63, a2);
    return v86;
  }
  return v7;
}
