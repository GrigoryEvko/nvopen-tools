// Function: sub_25332C0
// Address: 0x25332c0
//
__int64 __fastcall sub_25332C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v6; // rax
  unsigned int v7; // esi
  _QWORD *v8; // rdx
  __int64 *v9; // r11
  _BYTE *v10; // rbx
  __int64 v11; // r12
  unsigned int v12; // r13d
  unsigned int v13; // edi
  _QWORD *v14; // rcx
  __int64 v15; // rdx
  unsigned int v16; // edx
  _QWORD *v17; // r9
  int v18; // eax
  __m128i v19; // xmm0
  __m128i *v20; // r12
  __int64 *v21; // rbx
  __int64 v22; // r12
  __int64 v23; // r13
  unsigned __int8 *v24; // rdi
  int v25; // eax
  unsigned __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rax
  int v29; // edx
  __int64 v30; // rsi
  int v31; // ecx
  unsigned int v32; // edx
  __int64 v33; // rdi
  int v35; // r10d
  _QWORD *v36; // rdi
  unsigned int v37; // r13d
  int v38; // r8d
  __int64 v39; // rcx
  __int64 v40; // rcx
  __int64 *v41; // rbx
  __int64 v42; // r15
  __int64 i; // r14
  unsigned __int8 *v44; // rdi
  int v45; // eax
  unsigned __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rcx
  int v50; // r8d
  int v51; // r10d
  _QWORD *v52; // r8
  int v53; // r10d
  unsigned int v54; // r8d
  __int64 *v56; // [rsp+10h] [rbp-1300h]
  int v57; // [rsp+10h] [rbp-1300h]
  __int64 *v58; // [rsp+10h] [rbp-1300h]
  __int64 *v60; // [rsp+18h] [rbp-12F8h]
  __int64 *v61; // [rsp+18h] [rbp-12F8h]
  __int64 v62; // [rsp+20h] [rbp-12F0h] BYREF
  __int64 v63; // [rsp+28h] [rbp-12E8h]
  __int64 v64; // [rsp+30h] [rbp-12E0h]
  unsigned int v65; // [rsp+38h] [rbp-12D8h]
  __int64 v66; // [rsp+40h] [rbp-12D0h]
  _BYTE v67[16]; // [rsp+48h] [rbp-12C8h] BYREF
  void (__fastcall *v68)(unsigned __int64 *, unsigned __int64 *, __int64); // [rsp+58h] [rbp-12B8h]
  __int64 v69; // [rsp+60h] [rbp-12B0h]
  _BYTE v70[16]; // [rsp+68h] [rbp-12A8h] BYREF
  _QWORD *v71; // [rsp+78h] [rbp-1298h]
  __int64 v72; // [rsp+80h] [rbp-1290h]
  __int64 v73; // [rsp+88h] [rbp-1288h]
  __int64 *v74; // [rsp+90h] [rbp-1280h]
  __int64 v75; // [rsp+98h] [rbp-1278h]
  __m128i v76; // [rsp+A0h] [rbp-1270h] BYREF
  __int64 v77; // [rsp+B0h] [rbp-1260h]
  _BYTE v78[16]; // [rsp+B8h] [rbp-1258h] BYREF
  void (__fastcall *v79)(_BYTE *, _BYTE *, __int64); // [rsp+C8h] [rbp-1248h]
  __int64 v80; // [rsp+D0h] [rbp-1240h]
  __m128i v81; // [rsp+E0h] [rbp-1230h] BYREF
  __int64 v82; // [rsp+F0h] [rbp-1220h]
  void (__fastcall *v83)(unsigned __int64 *, unsigned __int64 *, __int64); // [rsp+F8h] [rbp-1218h]
  __int64 v84; // [rsp+100h] [rbp-1210h] BYREF
  _BYTE v85[8]; // [rsp+108h] [rbp-1208h] BYREF
  __int64 v86; // [rsp+110h] [rbp-1200h]
  _QWORD *v87; // [rsp+118h] [rbp-11F8h]
  __int64 v88; // [rsp+120h] [rbp-11F0h]
  __int64 v89; // [rsp+128h] [rbp-11E8h]
  _QWORD v90[2]; // [rsp+130h] [rbp-11E0h] BYREF
  __m128i v91; // [rsp+140h] [rbp-11D0h]
  __int64 v92; // [rsp+150h] [rbp-11C0h]
  _BYTE v93[16]; // [rsp+158h] [rbp-11B8h] BYREF
  void (__fastcall *v94)(_BYTE *, _BYTE *, __int64); // [rsp+168h] [rbp-11A8h]
  __int64 v95; // [rsp+170h] [rbp-11A0h]
  _QWORD v96[16]; // [rsp+180h] [rbp-1190h] BYREF
  _BYTE v97[192]; // [rsp+200h] [rbp-1110h] BYREF
  __int64 *v98; // [rsp+2C0h] [rbp-1050h]
  int v99; // [rsp+2C8h] [rbp-1048h]

  v66 = 0x101010000LL;
  v96[0] = &unk_438A674;
  v73 = a3;
  v96[1] = &unk_438A67B;
  v96[2] = &unk_438A676;
  LOBYTE(v66) = a5;
  v96[3] = &unk_438A67A;
  v68 = 0;
  v96[4] = &unk_438A679;
  v71 = 0;
  v96[5] = &unk_438A670;
  v74 = 0;
  BYTE4(v75) = 0;
  v76.m128i_i64[0] = 0;
  v77 = 0;
  v79 = 0;
  v96[6] = &unk_438A667;
  v62 = 0;
  v96[7] = &unk_438A668;
  v65 = 32;
  v96[8] = &unk_438A65D;
  v96[9] = &unk_438A66B;
  v96[10] = &unk_438A660;
  v96[11] = &unk_438A671;
  v96[12] = &unk_438A661;
  v96[13] = &unk_438A662;
  v96[14] = &unk_438A677;
  v96[15] = &unk_438A678;
  v6 = sub_C7D670(256, 8);
  v64 = 0;
  v63 = v6;
  v7 = 32;
  v8 = (_QWORD *)v6;
  if ( v6 != v6 + 256 )
  {
    do
    {
      if ( v8 )
        *v8 = -4096;
      ++v8;
    }
    while ( (_QWORD *)(v6 + 256) != v8 );
  }
  v9 = &v62;
  v10 = v96;
  while ( 1 )
  {
    if ( !v7 )
    {
      ++v62;
LABEL_11:
      v56 = v9;
      sub_2531380((__int64)v9, 2 * v7);
      if ( !v65 )
        goto LABEL_106;
      v9 = v56;
      v16 = (v65 - 1) & (((unsigned int)*(_QWORD *)v10 >> 9) ^ ((unsigned int)*(_QWORD *)v10 >> 4));
      v17 = (_QWORD *)(v63 + 8LL * v16);
      v11 = *v17;
      v18 = v64 + 1;
      if ( *(_QWORD *)v10 != *v17 )
      {
        v51 = 1;
        v52 = 0;
        while ( v11 != -4096 )
        {
          if ( v11 == -8192 && !v52 )
            v52 = v17;
          v16 = (v65 - 1) & (v16 + v51);
          v17 = (_QWORD *)(v63 + 8LL * v16);
          v11 = *v17;
          if ( *(_QWORD *)v10 == *v17 )
            goto LABEL_13;
          ++v51;
        }
        v11 = *(_QWORD *)v10;
        if ( v52 )
          v17 = v52;
      }
      goto LABEL_13;
    }
    v11 = *(_QWORD *)v10;
    v12 = ((unsigned int)*(_QWORD *)v10 >> 9) ^ ((unsigned int)*(_QWORD *)v10 >> 4);
    v13 = (v7 - 1) & v12;
    v14 = (_QWORD *)(v6 + 8LL * v13);
    v15 = *v14;
    if ( *v14 != *(_QWORD *)v10 )
      break;
LABEL_7:
    v10 += 8;
    if ( v10 == v97 )
      goto LABEL_16;
LABEL_8:
    v6 = v63;
    v7 = v65;
  }
  v35 = 1;
  v17 = 0;
  while ( v15 != -4096 )
  {
    if ( v15 != -8192 || v17 )
      v14 = v17;
    v13 = (v7 - 1) & (v35 + v13);
    v15 = *(_QWORD *)(v6 + 8LL * v13);
    if ( v11 == v15 )
      goto LABEL_7;
    ++v35;
    v17 = v14;
    v14 = (_QWORD *)(v6 + 8LL * v13);
  }
  if ( !v17 )
    v17 = v14;
  ++v62;
  v18 = v64 + 1;
  if ( 4 * ((int)v64 + 1) >= 3 * v7 )
    goto LABEL_11;
  if ( v7 - (v18 + HIDWORD(v64)) <= v7 >> 3 )
  {
    v58 = v9;
    sub_2531380((__int64)v9, v7);
    if ( !v65 )
    {
LABEL_106:
      LODWORD(v64) = v64 + 1;
      BUG();
    }
    v36 = 0;
    v9 = v58;
    v37 = (v65 - 1) & v12;
    v38 = 1;
    v17 = (_QWORD *)(v63 + 8LL * v37);
    v39 = *v17;
    v18 = v64 + 1;
    if ( v11 != *v17 )
    {
      while ( v39 != -4096 )
      {
        if ( !v36 && v39 == -8192 )
          v36 = v17;
        v53 = v38 + 1;
        v54 = (v65 - 1) & (v37 + v38);
        v17 = (_QWORD *)(v63 + 8LL * v54);
        v37 = v54;
        v39 = *v17;
        if ( v11 == *v17 )
          goto LABEL_13;
        v38 = v53;
      }
      if ( v36 )
        v17 = v36;
    }
  }
LABEL_13:
  LODWORD(v64) = v18;
  if ( *v17 != -4096 )
    --HIDWORD(v64);
  v10 += 8;
  *v17 = v11;
  if ( v10 != v97 )
    goto LABEL_8;
LABEL_16:
  v74 = v9;
  BYTE4(v66) = 0;
  v81.m128i_i32[0] = v66;
  v81.m128i_i16[2] = __PAIR16__(BYTE5(v66), 0);
  v83 = 0;
  if ( v68 )
  {
    v68(&v81.m128i_u64[1], (unsigned __int64 *)v67, 2);
    v84 = v69;
    v83 = v68;
  }
  v87 = 0;
  if ( v71 )
  {
    ((void (__fastcall *)(_BYTE *, _BYTE *, __int64))v71)(v85, v70, 2);
    v88 = v72;
    v87 = v71;
  }
  v19 = _mm_loadu_si128(&v76);
  v94 = 0;
  v89 = v73;
  v91 = v19;
  v90[0] = v74;
  v90[1] = v75;
  v92 = v77;
  if ( v79 )
  {
    v79(v93, v78, 2);
    v95 = v80;
    v94 = v79;
  }
  v20 = &v81;
  sub_250EFA0((__int64)v96, a2, a1, &v81);
  if ( v94 )
    v94(v93, v93, 3);
  if ( v87 )
    ((void (__fastcall *)(_BYTE *, _BYTE *, __int64))v87)(v85, v85, 3);
  if ( v83 )
    v83(&v81.m128i_u64[1], &v81.m128i_u64[1], 3);
  v21 = *(__int64 **)(a2 + 32);
  v60 = &v21[*(unsigned int *)(a2 + 40)];
  if ( v21 != v60 )
  {
    while ( 1 )
    {
LABEL_32:
      v22 = *v21;
      if ( !sub_B2FC80(*v21) )
        sub_B2FC00((_BYTE *)v22);
      if ( BYTE4(v66) && (*(_BYTE *)(v22 + 32) & 0xFu) - 7 <= 1 )
      {
        v23 = *(_QWORD *)(v22 + 16);
        if ( !v23 )
          goto LABEL_31;
        while ( 1 )
        {
          v24 = *(unsigned __int8 **)(v23 + 24);
          v25 = *v24;
          if ( (unsigned __int8)v25 <= 0x1Cu )
            break;
          v26 = (unsigned int)(v25 - 34);
          if ( (unsigned __int8)v26 > 0x33u )
            break;
          v27 = 0x8000000000041LL;
          if ( !_bittest64(&v27, v26) )
            break;
          if ( (unsigned __int8 *)v23 != v24 - 32 )
            break;
          v28 = sub_B491C0((__int64)v24);
          v29 = *(_DWORD *)(a2 + 24);
          v30 = *(_QWORD *)(a2 + 8);
          if ( !v29 )
            break;
          v31 = v29 - 1;
          v32 = (v29 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
          v33 = *(_QWORD *)(v30 + 8LL * v32);
          if ( v28 != v33 )
          {
            v50 = 1;
            while ( v33 != -4096 )
            {
              v32 = v31 & (v50 + v32);
              v33 = *(_QWORD *)(v30 + 8LL * v32);
              if ( v28 == v33 )
                goto LABEL_43;
              ++v50;
            }
            break;
          }
LABEL_43:
          v23 = *(_QWORD *)(v23 + 8);
          if ( !v23 )
          {
            if ( v60 != ++v21 )
              goto LABEL_32;
            goto LABEL_45;
          }
        }
      }
      sub_252C210((__int64)v96, v22);
LABEL_31:
      if ( v60 == ++v21 )
      {
LABEL_45:
        v20 = &v81;
        break;
      }
    }
  }
  v57 = sub_2531E70((__int64)v96);
  if ( !v57 )
  {
    v81.m128i_i64[0] = 0;
    v81.m128i_i64[1] = (__int64)&v84;
    v82 = 2;
    LODWORD(v83) = 0;
    BYTE4(v83) = 1;
    v86 = 0;
    v87 = v90;
    v88 = 2;
    LODWORD(v89) = 0;
    BYTE4(v89) = 1;
    sub_AE6EC0((__int64)&v81, (__int64)&unk_4F82408);
    v41 = v98;
    v61 = &v98[v99];
    if ( v98 != v61 )
    {
      do
      {
        v42 = *v41;
        sub_BBE020(a4, *v41, (__int64)&v81, v40);
        for ( i = *(_QWORD *)(v42 + 16); i; i = *(_QWORD *)(i + 8) )
        {
          v44 = *(unsigned __int8 **)(i + 24);
          v45 = *v44;
          if ( (unsigned __int8)v45 > 0x1Cu )
          {
            v46 = (unsigned int)(v45 - 34);
            if ( (unsigned __int8)v46 <= 0x33u )
            {
              v40 = 0x8000000000041LL;
              if ( _bittest64(&v40, v46) )
              {
                v47 = *((_QWORD *)v44 - 4);
                if ( v47 )
                {
                  if ( !*(_BYTE *)v47 )
                  {
                    v40 = *((_QWORD *)v44 + 10);
                    if ( *(_QWORD *)(v47 + 24) == v40 && v42 == v47 )
                    {
                      v48 = sub_B43CB0((__int64)v44);
                      sub_BBE020(a4, v48, (__int64)&v81, v49);
                    }
                  }
                }
              }
            }
          }
        }
        ++v41;
      }
      while ( v61 != v41 );
    }
    if ( !BYTE4(v89) )
      _libc_free((unsigned __int64)v87);
    if ( !BYTE4(v83) )
      _libc_free(v81.m128i_u64[1]);
  }
  sub_250D880((__int64)v96);
  sub_C7D6A0(v63, 8LL * v65, 8);
  if ( v79 )
    v79(v78, v78, 3);
  if ( v71 )
    ((void (__fastcall *)(_BYTE *, _BYTE *, __int64))v71)(v70, v70, 3);
  if ( v68 )
    v68((unsigned __int64 *)v67, (unsigned __int64 *)v67, 3);
  LOBYTE(v20) = v57 == 0;
  return (unsigned int)v20;
}
