// Function: sub_A07720
// Address: 0xa07720
//
__int64 __fastcall sub_A07720(__int64 a1, __m128i *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 m128i_i64; // r13
  __int64 *v11; // r15
  __int64 *v12; // rsi
  __int64 v13; // rbx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // r13
  char v18; // al
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rdx
  const char *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  char v24; // al
  __int64 v25; // rcx
  __int64 v26; // rbx
  __int64 v27; // rcx
  char v28; // dl
  char v29; // al
  char v30; // dl
  const void *v31; // r8
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rax
  void *v36; // r9
  signed __int64 v37; // rdx
  __int64 *v38; // r13
  unsigned int *v39; // r15
  unsigned int *v40; // rbx
  _BYTE *v41; // rsi
  _BYTE *v42; // rdx
  const __m128i *v43; // rdi
  __int64 v44; // r11
  unsigned __int64 v45; // rsi
  const __m128i *v46; // rax
  const __m128i *v47; // rcx
  __m128i *v48; // r8
  __int64 v49; // r9
  signed __int64 v50; // rbx
  __int64 v51; // rax
  __m128i *v52; // rdx
  _BYTE *v53; // rcx
  __int64 v54; // r8
  unsigned int *v55; // rbx
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rdx
  char v59; // cl
  __int64 *v60; // r13
  __int64 v61; // rax
  char v62; // dl
  _BYTE *v63; // rsi
  __int64 v64; // rax
  __int64 v65; // rbx
  unsigned __int64 v66; // r12
  __int64 v67; // r13
  _BYTE *v68; // rax
  void *v69; // rax
  __int64 v70; // rsi
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rdi
  __int64 v75; // [rsp+8h] [rbp-2D8h]
  __int64 v76; // [rsp+8h] [rbp-2D8h]
  __int64 v77; // [rsp+8h] [rbp-2D8h]
  void *v78; // [rsp+8h] [rbp-2D8h]
  int v79; // [rsp+24h] [rbp-2BCh]
  unsigned __int32 v80; // [rsp+28h] [rbp-2B8h]
  unsigned int v81; // [rsp+28h] [rbp-2B8h]
  __int64 v82; // [rsp+28h] [rbp-2B8h]
  __int64 v83; // [rsp+30h] [rbp-2B0h]
  __int64 v84; // [rsp+30h] [rbp-2B0h]
  __int64 v85; // [rsp+30h] [rbp-2B0h]
  const void *v86; // [rsp+30h] [rbp-2B0h]
  __int64 v87; // [rsp+30h] [rbp-2B0h]
  unsigned __int32 v88; // [rsp+38h] [rbp-2A8h]
  __int64 v89; // [rsp+38h] [rbp-2A8h]
  __int64 v90; // [rsp+38h] [rbp-2A8h]
  __m128i *v91; // [rsp+38h] [rbp-2A8h]
  unsigned int v92; // [rsp+48h] [rbp-298h]
  unsigned int v93; // [rsp+4Ch] [rbp-294h]
  __m128i *v94; // [rsp+50h] [rbp-290h] BYREF
  char v95; // [rsp+58h] [rbp-288h]
  unsigned __int64 v96; // [rsp+60h] [rbp-280h] BYREF
  unsigned __int64 v97; // [rsp+68h] [rbp-278h]
  unsigned __int64 v98; // [rsp+70h] [rbp-270h] BYREF
  __int64 v99; // [rsp+78h] [rbp-268h]
  __int64 v100; // [rsp+80h] [rbp-260h]
  _BYTE v101[9]; // [rsp+88h] [rbp-258h] BYREF
  char v102; // [rsp+91h] [rbp-24Fh]
  unsigned int *v103; // [rsp+A0h] [rbp-240h] BYREF
  __int64 v104; // [rsp+A8h] [rbp-238h]
  _BYTE v105[560]; // [rsp+B0h] [rbp-230h] BYREF

  v6 = a2[15].m128i_i64[0];
  a2[23] = _mm_loadu_si128((const __m128i *)v6);
  a2[24] = _mm_loadu_si128((const __m128i *)(v6 + 16));
  a2[25].m128i_i32[0] = *(_DWORD *)(v6 + 32);
  a2[25].m128i_i32[1] = *(_DWORD *)(v6 + 36);
  sub_A020B0((__int64)&a2[25].m128i_i64[1], (char **)(v6 + 40), a3, a4);
  sub_A05260((__int64)a2[27].m128i_i64, v6 + 64, v7);
  v9 = *(_QWORD *)(v6 + 336);
  a2[47].m128i_i64[1] = 0;
  a2[44].m128i_i64[0] = v9;
  v103 = (unsigned int *)v105;
  v104 = 0x4000000000LL;
  m128i_i64 = (__int64)a2[23].m128i_i64;
  v11 = (__int64 *)&v98;
  do
  {
LABEL_2:
    v12 = (__int64 *)m128i_i64;
    v13 = a2[24].m128i_i64[0];
    v88 = a2[25].m128i_u32[0];
    sub_9CEFB0((__int64)v11, m128i_i64, 1, v8);
    if ( (v99 & 1) != 0 )
    {
      LOBYTE(v99) = v99 & 0xFD;
      v15 = v98;
      v98 = 0;
      v96 = v15 | 1;
      v16 = v15 & 0xFFFFFFFFFFFFFFFELL;
      if ( v16 )
        goto LABEL_4;
    }
    else
    {
      v96 = 1;
      v92 = HIDWORD(v98);
      v93 = v98;
    }
    if ( v93 == 2 )
    {
      v17 = (__int64)v11;
LABEL_20:
      v102 = 1;
      v21 = "Malformed block";
LABEL_21:
      v12 = (__int64 *)v17;
      v98 = (unsigned __int64)v21;
      v101[8] = 3;
      sub_A01DB0((__int64 *)&v96, v17);
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v96 & 0xFFFFFFFFFFFFFFFELL;
      goto LABEL_5;
    }
    if ( v93 <= 2 )
    {
      v17 = (__int64)v11;
      if ( v93 )
      {
        v18 = *(_BYTE *)(a1 + 8);
        *(_BYTE *)a1 = 1;
        *(_BYTE *)(a1 + 8) = v18 & 0xFC | 2;
        goto LABEL_5;
      }
      goto LABEL_20;
    }
  }
  while ( v93 != 3 );
  v12 = (__int64 *)m128i_i64;
  v83 = a2[24].m128i_i64[0];
  v80 = a2[25].m128i_u32[0];
  sub_A4CAE0(v11, m128i_i64, v92);
  if ( (v99 & 1) != 0 )
  {
    LOBYTE(v99) = v99 & 0xFD;
    v19 = v98;
    v98 = 0;
    v96 = v19 | 1;
  }
  else
  {
    v96 = 1;
    v79 = v98;
  }
  v16 = v96 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v96 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_4;
  v20 = 8 * v83 - v80;
  switch ( v79 )
  {
    case 1:
    case 2:
    case 3:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
    case 22:
    case 23:
    case 24:
    case 25:
    case 26:
    case 27:
    case 28:
    case 29:
    case 30:
    case 31:
    case 32:
    case 33:
    case 34:
    case 37:
    case 40:
    case 41:
    case 44:
    case 45:
    case 47:
      v22 = a2[44].m128i_i64[1];
      if ( v22 != a2[45].m128i_i64[0] )
        a2[45].m128i_i64[0] = v22;
      v23 = a2[46].m128i_i64[0];
      if ( v23 != a2[46].m128i_i64[1] )
        a2[46].m128i_i64[1] = v23;
      v24 = *(_BYTE *)(a1 + 8);
      *(_BYTE *)a1 = 0;
      *(_BYTE *)(a1 + 8) = v24 & 0xFC | 2;
      goto LABEL_5;
    case 4:
      v12 = (__int64 *)m128i_i64;
      sub_9CDFE0(v11, m128i_i64, v20, v8);
      v16 = v98 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v98 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_4;
      v12 = (__int64 *)m128i_i64;
      LODWORD(v104) = 0;
      sub_A4B600(v11, m128i_i64, v92, &v103, 0);
      if ( (v99 & 1) != 0 )
        goto LABEL_96;
      v54 = (unsigned int)v104;
      v99 = 0;
      v55 = v103;
      v98 = (unsigned __int64)v101;
      v56 = (unsigned int)v104;
      v100 = 8;
      if ( (unsigned int)v104 > 8uLL )
      {
        v76 = (unsigned int)v104;
        sub_C8D290(v11, v101, (unsigned int)v104, 1);
        v54 = v76;
        v53 = (_BYTE *)(v98 + v99);
      }
      else
      {
        if ( !(8LL * (unsigned int)v104) )
          goto LABEL_70;
        v53 = v101;
      }
      v57 = 0;
      do
      {
        v53[v57] = *(_QWORD *)&v55[2 * v57];
        ++v57;
      }
      while ( v54 != v57 );
      v56 = v54 + v99;
LABEL_70:
      v58 = a2[25].m128i_u32[1];
      v99 = v56;
      sub_9C66D0((__int64)&v96, m128i_i64, v58, (__int64)v53);
      v59 = v97 & 1;
      LOBYTE(v97) = v97 & 0xFD;
      v95 = v59 | v95 & 0xFE | 2;
      if ( v59 )
      {
        v60 = (__int64 *)&v94;
        v12 = (__int64 *)&v94;
        v94 = (__m128i *)v96;
        sub_9C8CD0((__int64 *)&v96, (__int64 *)&v94);
        v61 = v96;
        *(_BYTE *)(a1 + 8) |= 3u;
        *(_QWORD *)a1 = v61 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v95 & 2) == 0 )
        {
          if ( (v95 & 1) == 0 || (v74 = (__int64)v94) == 0 )
          {
LABEL_73:
            if ( (_BYTE *)v98 != v101 )
              _libc_free(v98, v12);
            goto LABEL_5;
          }
LABEL_109:
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v74 + 8LL))(v74);
          goto LABEL_73;
        }
LABEL_115:
        sub_9CE230(v60);
      }
      LODWORD(v104) = 0;
      sub_A4B600(&v96, m128i_i64, (unsigned int)v96, &v103, 0);
      v62 = v97 & 1;
      LOBYTE(v97) = (2 * (v97 & 1)) | v97 & 0xFD;
      if ( v62 )
      {
        v12 = (__int64 *)&v96;
        sub_9C8CD0((__int64 *)&v94, (__int64 *)&v96);
        v73 = (__int64)v94;
        *(_BYTE *)(a1 + 8) |= 3u;
        *(_QWORD *)a1 = v73 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v97 & 2) != 0 )
          sub_9CE230(&v96);
        if ( (v97 & 1) == 0 )
          goto LABEL_73;
        v74 = v96;
        if ( !v96 )
          goto LABEL_73;
        goto LABEL_109;
      }
      v63 = (_BYTE *)v98;
      v81 = v104;
      v64 = sub_BA8E40(a2[16].m128i_i64[0], v98, v99);
      v8 = v81;
      v65 = v64;
      if ( v81 )
      {
        v75 = a1;
        v66 = 0;
        v82 = m128i_i64;
        v67 = 8LL * (unsigned int)v8;
        do
        {
          v68 = (_BYTE *)sub_A07560((__int64)a2, *(_QWORD *)&v103[v66 / 4]);
          v63 = v68;
          if ( v68 && (unsigned __int8)(*v68 - 5) >= 0x20u )
            v63 = 0;
          v66 += 8LL;
          sub_B979A0(v65, v63);
        }
        while ( v66 != v67 );
        m128i_i64 = v82;
        a1 = v75;
      }
      if ( (_BYTE *)v98 != v101 )
        _libc_free(v98, v63);
      goto LABEL_2;
    case 35:
      v12 = (__int64 *)m128i_i64;
      sub_9CDFE0(v11, m128i_i64, v20, v8);
      v16 = v98 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v98 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_4;
      v12 = (__int64 *)m128i_i64;
      v96 = 0;
      v97 = 0;
      LODWORD(v104) = 0;
      sub_A4B600(v11, m128i_i64, v92, &v103, &v96);
      if ( (v99 & 1) != 0 )
        goto LABEL_96;
      v43 = (const __m128i *)a2[44].m128i_i64[1];
      v44 = (__int64)v103;
      v45 = *v103;
      v46 = v43;
      if ( v45 > (a2[45].m128i_i64[1] - (__int64)v43) >> 4 )
      {
        v47 = (const __m128i *)a2[45].m128i_i64[0];
        v48 = 0;
        v49 = 16 * v45;
        v50 = (char *)v47 - (char *)v43;
        if ( *v103 )
        {
          v51 = sub_22077B0(16 * v45);
          v43 = (const __m128i *)a2[44].m128i_i64[1];
          v47 = (const __m128i *)a2[45].m128i_i64[0];
          v49 = 16 * v45;
          v48 = (__m128i *)v51;
          v46 = v43;
        }
        v52 = v48;
        if ( v47 != v43 )
        {
          do
          {
            if ( v52 )
              *v52 = _mm_loadu_si128(v46);
            ++v46;
            ++v52;
          }
          while ( v47 != v46 );
        }
        if ( v43 )
        {
          v85 = v49;
          v91 = v48;
          j_j___libc_free_0(v43, a2[45].m128i_i64[1] - (_QWORD)v43);
          v49 = v85;
          v48 = v91;
        }
        a2[44].m128i_i64[1] = (__int64)v48;
        v44 = (__int64)v103;
        a2[45].m128i_i64[0] = (__int64)v48->m128i_i64 + v50;
        a2[45].m128i_i64[1] = (__int64)v48->m128i_i64 + v49;
      }
      v12 = (__int64 *)a2;
      v94 = a2;
      sub_A04C00(
        v11,
        (__int64)a2,
        v44,
        (unsigned int)v104,
        v96,
        v97,
        (void (__fastcall *)(__int64, __int64))sub_A04390,
        (__int64)&v94);
      v16 = v98 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v98 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_4;
      goto LABEL_2;
    case 36:
      if ( !a2[47].m128i_i64[1] )
        a2[47].m128i_i64[1] = 8 * v13 - v88;
      goto LABEL_2;
    case 38:
      v12 = (__int64 *)m128i_i64;
      sub_9CDFE0(v11, m128i_i64, v20, v8);
      v16 = v98 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v98 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_4;
      v12 = (__int64 *)m128i_i64;
      LODWORD(v104) = 0;
      sub_A4B600(v11, m128i_i64, v92, &v103, 0);
      if ( (v99 & 1) != 0 )
      {
LABEL_96:
        v16 = v98 & 0xFFFFFFFFFFFFFFFELL;
LABEL_4:
        *(_BYTE *)(a1 + 8) |= 3u;
        *(_QWORD *)a1 = v16;
        goto LABEL_5;
      }
      if ( (_DWORD)v104 != 2 )
      {
        v102 = 1;
        v17 = (__int64)v11;
        v21 = "Invalid record";
        goto LABEL_21;
      }
      v25 = a2[24].m128i_i64[0];
      v12 = (__int64 *)m128i_i64;
      v26 = 8 * v25 - a2[25].m128i_u32[0];
      sub_9CDFE0(v11, m128i_i64, v26 + *(_QWORD *)v103 + (*((_QWORD *)v103 + 1) << 32), v25);
      v16 = v98 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v98 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        goto LABEL_4;
      v12 = (__int64 *)m128i_i64;
      sub_9CEFB0((__int64)&v96, m128i_i64, 1, v27);
      v28 = v97 & 1;
      v29 = (2 * (v97 & 1)) | v97 & 0xFD;
      LOBYTE(v97) = v29;
      if ( v28 )
      {
        *(_BYTE *)(a1 + 8) |= 3u;
        LOBYTE(v97) = v29 & 0xFD;
        v72 = v96;
        v96 = 0;
        *(_QWORD *)a1 = v72 & 0xFFFFFFFFFFFFFFFELL;
        goto LABEL_5;
      }
      LODWORD(v104) = 0;
      sub_A4B600(v11, m128i_i64, HIDWORD(v96), &v103, 0);
      v30 = v99 & 1;
      LOBYTE(v99) = (2 * (v99 & 1)) | v99 & 0xFD;
      if ( !v30 )
      {
        v31 = (const void *)a2[46].m128i_i64[0];
        v32 = a2[47].m128i_i64[0];
        v98 = v26;
        v33 = (unsigned int)v104;
        if ( (unsigned int)v104 <= (unsigned __int64)((v32 - (__int64)v31) >> 3) )
          goto LABEL_40;
        v34 = 8LL * (unsigned int)v104;
        v89 = a2[46].m128i_i64[1] - (_QWORD)v31;
        if ( (_DWORD)v104 )
        {
          v84 = 8LL * (unsigned int)v104;
          v35 = sub_22077B0(v84);
          v31 = (const void *)a2[46].m128i_i64[0];
          v34 = v84;
          v36 = (void *)v35;
          v37 = a2[46].m128i_i64[1] - (_QWORD)v31;
        }
        else
        {
          v37 = a2[46].m128i_i64[1] - (_QWORD)v31;
          v36 = 0;
        }
        if ( v37 > 0 )
        {
          v77 = v34;
          v86 = v31;
          v69 = memmove(v36, v31, v37);
          v31 = v86;
          v34 = v77;
          v36 = v69;
          v70 = a2[47].m128i_i64[0] - (_QWORD)v86;
LABEL_95:
          v78 = v36;
          v87 = v34;
          j_j___libc_free_0(v31, v70);
          v36 = v78;
          v34 = v87;
          goto LABEL_39;
        }
        if ( v31 )
        {
          v70 = a2[47].m128i_i64[0] - (_QWORD)v31;
          goto LABEL_95;
        }
LABEL_39:
        a2[46].m128i_i64[0] = (__int64)v36;
        a2[47].m128i_i64[0] = (__int64)v36 + v34;
        v33 = (unsigned int)v104;
        a2[46].m128i_i64[1] = (__int64)v36 + v89;
LABEL_40:
        v8 = (__int64)v103;
        if ( &v103[2 * v33] != v103 )
        {
          v90 = m128i_i64;
          v38 = v11;
          v39 = &v103[2 * v33];
          v40 = v103;
          do
          {
            v42 = (_BYTE *)(*(_QWORD *)v40 + v98);
            v98 = (unsigned __int64)v42;
            v41 = (_BYTE *)a2[46].m128i_i64[1];
            if ( v41 == (_BYTE *)a2[47].m128i_i64[0] )
            {
              sub_9CA200((__int64)a2[46].m128i_i64, v41, v38);
            }
            else
            {
              if ( v41 )
              {
                *(_QWORD *)v41 = v42;
                v41 = (_BYTE *)a2[46].m128i_i64[1];
              }
              a2[46].m128i_i64[1] = (__int64)(v41 + 8);
            }
            v40 += 2;
          }
          while ( v39 != v40 );
          v11 = v38;
          m128i_i64 = v90;
        }
        if ( (v97 & 2) != 0 )
          goto LABEL_97;
        if ( (v97 & 1) != 0 && v96 )
          (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v96 + 8LL))(v96);
        goto LABEL_2;
      }
      v12 = v11;
      v60 = v11;
      sub_9C8CD0((__int64 *)&v94, v11);
      v71 = (__int64)v94;
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v71 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v99 & 2) != 0 )
        goto LABEL_115;
      if ( (v99 & 1) != 0 && v98 )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v98 + 8LL))(v98);
      if ( (v97 & 2) != 0 )
LABEL_97:
        sub_9CEF10(&v96);
      if ( (v97 & 1) != 0 && v96 )
        (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v96 + 8LL))(v96);
LABEL_5:
      if ( v103 != (unsigned int *)v105 )
        _libc_free(v103, v12);
      return a1;
    case 39:
      v102 = 1;
      v17 = (__int64)v11;
      v21 = "Corrupted Metadata block";
      goto LABEL_21;
    default:
      goto LABEL_2;
  }
}
