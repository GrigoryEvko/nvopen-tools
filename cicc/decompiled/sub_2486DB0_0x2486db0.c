// Function: sub_2486DB0
// Address: 0x2486db0
//
void __fastcall sub_2486DB0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r14
  int v5; // eax
  unsigned __int64 *v6; // rax
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rcx
  int v11; // eax
  _BYTE *v12; // rsi
  int v13; // ecx
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  char v16; // r10
  __int64 v17; // r8
  __int64 v18; // rdi
  char *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  char v22; // dl
  _QWORD *v23; // rcx
  char v24; // cl
  __int64 v25; // rdi
  __int64 v26; // rax
  __m128i v27; // xmm1
  char v28; // dl
  char v29; // al
  __m128i *v30; // rsi
  _QWORD *v31; // rcx
  char v32; // al
  __m128i *v33; // rsi
  __m128i *v34; // rcx
  __m128i v35; // xmm5
  __m128i *v36; // rax
  _QWORD *v37; // rsi
  __m128i v38; // xmm7
  __m128i v39; // xmm7
  __int64 v40; // [rsp+8h] [rbp-268h]
  __int64 v41; // [rsp+10h] [rbp-260h]
  __int64 v42; // [rsp+18h] [rbp-258h]
  __int64 v43; // [rsp+20h] [rbp-250h]
  __int64 v44; // [rsp+28h] [rbp-248h]
  __int64 v45; // [rsp+30h] [rbp-240h]
  __int64 v46[2]; // [rsp+40h] [rbp-230h] BYREF
  __int64 v47; // [rsp+50h] [rbp-220h] BYREF
  unsigned __int64 v48[2]; // [rsp+60h] [rbp-210h] BYREF
  __int64 v49; // [rsp+70h] [rbp-200h] BYREF
  __int64 v50[2]; // [rsp+80h] [rbp-1F0h] BYREF
  _BYTE v51[16]; // [rsp+90h] [rbp-1E0h] BYREF
  _QWORD *v52; // [rsp+A0h] [rbp-1D0h] BYREF
  int v53; // [rsp+A8h] [rbp-1C8h]
  _QWORD v54[2]; // [rsp+B0h] [rbp-1C0h] BYREF
  _QWORD v55[4]; // [rsp+C0h] [rbp-1B0h] BYREF
  char v56; // [rsp+E0h] [rbp-190h]
  char v57; // [rsp+E1h] [rbp-18Fh]
  __m128i v58; // [rsp+F0h] [rbp-180h] BYREF
  __m128i v59; // [rsp+100h] [rbp-170h] BYREF
  __int64 v60; // [rsp+110h] [rbp-160h]
  _QWORD v61[4]; // [rsp+120h] [rbp-150h] BYREF
  __int16 v62; // [rsp+140h] [rbp-130h]
  __m128i v63; // [rsp+150h] [rbp-120h] BYREF
  __m128i v64; // [rsp+160h] [rbp-110h] BYREF
  __int64 v65; // [rsp+170h] [rbp-100h]
  _QWORD v66[2]; // [rsp+180h] [rbp-F0h] BYREF
  __int64 v67; // [rsp+1A0h] [rbp-D0h]
  __m128i v68; // [rsp+1B0h] [rbp-C0h] BYREF
  __m128i v69; // [rsp+1C0h] [rbp-B0h] BYREF
  __int64 v70; // [rsp+1D0h] [rbp-A0h]
  __m128i v71; // [rsp+1E0h] [rbp-90h] BYREF
  __m128i v72; // [rsp+1F0h] [rbp-80h] BYREF
  __int64 v73; // [rsp+200h] [rbp-70h]
  __m128i v74; // [rsp+210h] [rbp-60h] BYREF
  __m128i v75; // [rsp+220h] [rbp-50h]
  __int64 v76; // [rsp+230h] [rbp-40h]

  v5 = *((_DWORD *)a2 + 2);
  if ( v5 == 13 )
  {
    if ( !unk_4FEBF28 )
      return;
  }
  else if ( v5 == 15
         && (unk_4FEBE48 || LOBYTE(qword_4FEBCE0[17]) && (*(_QWORD *)(*a1 + 48) || (*(_BYTE *)(*a1 + 32) & 0xF) == 1)) )
  {
    return;
  }
  v6 = (unsigned __int64 *)a1[1];
  v7 = *v6;
  if ( *v6 <= 9 )
  {
    v52 = v54;
    sub_2240A50((__int64 *)&v52, 1u, 0);
    v12 = v52;
LABEL_16:
    *v12 = v7 + 48;
    goto LABEL_17;
  }
  if ( v7 <= 0x63 )
  {
    v52 = v54;
    sub_2240A50((__int64 *)&v52, 2u, 0);
    v12 = v52;
  }
  else
  {
    if ( v7 <= 0x3E7 )
    {
      v9 = 3;
    }
    else if ( v7 <= 0x270F )
    {
      v9 = 4;
    }
    else
    {
      v8 = *v6;
      LODWORD(v9) = 1;
      while ( 1 )
      {
        v10 = v8;
        v11 = v9;
        v9 = (unsigned int)(v9 + 4);
        v8 /= 0x2710u;
        if ( v10 <= 0x1869F )
          break;
        if ( v10 <= 0xF423F )
        {
          v9 = (unsigned int)(v11 + 5);
          v52 = v54;
          goto LABEL_13;
        }
        if ( v10 <= (unsigned __int64)&loc_98967F )
        {
          v9 = (unsigned int)(v11 + 6);
          break;
        }
        if ( v10 <= 0x5F5E0FF )
        {
          v9 = (unsigned int)(v11 + 7);
          break;
        }
      }
    }
    v52 = v54;
LABEL_13:
    sub_2240A50((__int64 *)&v52, v9, 0);
    v12 = v52;
    v13 = v53 - 1;
    do
    {
      v14 = v7
          - 20 * (v7 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v7 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
      v15 = v7;
      v7 /= 0x64u;
      v16 = a00010203040506_0[2 * v14 + 1];
      LOBYTE(v14) = a00010203040506_0[2 * v14];
      v12[v13] = v16;
      v17 = (unsigned int)(v13 - 1);
      v13 -= 2;
      v12[v17] = v14;
    }
    while ( v15 > 0x270F );
    if ( v15 <= 0x3E7 )
      goto LABEL_16;
  }
  v12[1] = a00010203040506_0[2 * v7 + 1];
  *v12 = a00010203040506_0[2 * v7];
LABEL_17:
  v18 = *a1;
  v71.m128i_i64[0] = (__int64)&v52;
  LOWORD(v73) = 260;
  v66[0] = " Hash = ";
  LOWORD(v67) = 259;
  v19 = (char *)sub_BD5D20(v18);
  if ( v19 )
  {
    v50[0] = (__int64)v51;
    sub_2485610(v50, v19, (__int64)&v19[v20]);
  }
  else
  {
    v51[0] = 0;
    v50[0] = (__int64)v51;
    v50[1] = 0;
  }
  v21 = *a2;
  v61[0] = v50;
  v62 = 260;
  v57 = 1;
  v55[0] = " ";
  v56 = 3;
  (*(void (__fastcall **)(unsigned __int64 *, __int64 *))(v21 + 24))(v48, a2);
  v22 = v56;
  if ( !v56 )
  {
    LOWORD(v60) = 256;
    goto LABEL_24;
  }
  if ( v56 != 1 )
  {
    if ( v57 == 1 )
    {
      v23 = (_QWORD *)v55[0];
      v45 = v55[1];
    }
    else
    {
      v23 = v55;
      v22 = 2;
    }
    v59.m128i_i64[0] = (__int64)v23;
    v24 = v62;
    v58.m128i_i64[0] = (__int64)v48;
    v59.m128i_i64[1] = v45;
    LOBYTE(v60) = 4;
    BYTE1(v60) = v22;
    if ( !(_BYTE)v62 )
      goto LABEL_24;
    if ( (_BYTE)v62 != 1 )
    {
      v36 = &v58;
      v28 = 2;
      goto LABEL_70;
    }
    goto LABEL_45;
  }
  v24 = v62;
  v58.m128i_i64[0] = (__int64)v48;
  LOWORD(v60) = 260;
  if ( !(_BYTE)v62 )
  {
LABEL_24:
    LOWORD(v65) = 256;
LABEL_25:
    LOWORD(v70) = 256;
    goto LABEL_26;
  }
  if ( (_BYTE)v62 != 1 )
  {
    v2 = v58.m128i_i64[1];
    v36 = (__m128i *)v48;
    v28 = 4;
LABEL_70:
    if ( HIBYTE(v62) == 1 )
    {
      v37 = (_QWORD *)v61[0];
      v44 = v61[1];
    }
    else
    {
      v37 = v61;
      v24 = 2;
    }
    v63.m128i_i64[0] = (__int64)v36;
    v63.m128i_i64[1] = v2;
    v64.m128i_i64[1] = v44;
    v29 = v67;
    v64.m128i_i64[0] = (__int64)v37;
    LOBYTE(v65) = v28;
    BYTE1(v65) = v24;
    if ( !(_BYTE)v67 )
      goto LABEL_25;
    goto LABEL_46;
  }
LABEL_45:
  v27 = _mm_loadu_si128(&v59);
  v28 = v60;
  v63 = _mm_loadu_si128(&v58);
  v65 = v60;
  v64 = v27;
  v29 = v67;
  if ( !(_BYTE)v67 )
    goto LABEL_25;
LABEL_46:
  if ( v29 == 1 )
  {
    v39 = _mm_loadu_si128(&v64);
    v28 = v65;
    v68 = _mm_loadu_si128(&v63);
    v70 = v65;
    v69 = v39;
    if ( !(_BYTE)v65 )
      goto LABEL_26;
  }
  else
  {
    if ( BYTE1(v65) == 1 )
    {
      v43 = v63.m128i_i64[1];
      v30 = (__m128i *)v63.m128i_i64[0];
    }
    else
    {
      v30 = &v63;
      v28 = 2;
    }
    if ( BYTE1(v67) == 1 )
    {
      v42 = v66[1];
      v31 = (_QWORD *)v66[0];
    }
    else
    {
      v31 = v66;
      v29 = 2;
    }
    v68.m128i_i64[0] = (__int64)v30;
    v69.m128i_i64[0] = (__int64)v31;
    v68.m128i_i64[1] = v43;
    LOBYTE(v70) = v28;
    v69.m128i_i64[1] = v42;
    BYTE1(v70) = v29;
  }
  v32 = v73;
  if ( (_BYTE)v73 )
  {
    if ( v28 == 1 )
    {
      v35 = _mm_loadu_si128(&v72);
      v74 = _mm_loadu_si128(&v71);
      v76 = v73;
      v75 = v35;
    }
    else if ( (_BYTE)v73 == 1 )
    {
      v38 = _mm_loadu_si128(&v69);
      v74 = _mm_loadu_si128(&v68);
      v76 = v70;
      v75 = v38;
    }
    else
    {
      if ( BYTE1(v70) == 1 )
      {
        v41 = v68.m128i_i64[1];
        v33 = (__m128i *)v68.m128i_i64[0];
      }
      else
      {
        v33 = &v68;
        v28 = 2;
      }
      if ( BYTE1(v73) == 1 )
      {
        v40 = v71.m128i_i64[1];
        v34 = (__m128i *)v71.m128i_i64[0];
      }
      else
      {
        v32 = 2;
        v34 = &v71;
      }
      v74.m128i_i64[0] = (__int64)v33;
      v75.m128i_i64[0] = (__int64)v34;
      v74.m128i_i64[1] = v41;
      LOBYTE(v76) = v28;
      v75.m128i_i64[1] = v40;
      BYTE1(v76) = v32;
    }
    goto LABEL_27;
  }
LABEL_26:
  LOWORD(v76) = 256;
LABEL_27:
  sub_CA0F50(v46, (void **)&v74);
  if ( (__int64 *)v48[0] != &v49 )
    j_j___libc_free_0(v48[0]);
  if ( (_BYTE *)v50[0] != v51 )
    j_j___libc_free_0(v50[0]);
  if ( v52 != v54 )
    j_j___libc_free_0((unsigned __int64)v52);
  v25 = a1[2];
  v74.m128i_i64[0] = (__int64)v46;
  LOWORD(v76) = 260;
  v26 = *(_QWORD *)(a1[3] + 168);
  v71.m128i_i64[1] = 0x100000017LL;
  v72.m128i_i64[1] = (__int64)&v74;
  v72.m128i_i64[0] = v26;
  v71.m128i_i64[0] = (__int64)&unk_49D9CA8;
  sub_B6EB20(v25, (__int64)&v71);
  if ( (__int64 *)v46[0] != &v47 )
    j_j___libc_free_0(v46[0]);
}
