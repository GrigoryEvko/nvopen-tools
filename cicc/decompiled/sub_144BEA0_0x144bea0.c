// Function: sub_144BEA0
// Address: 0x144bea0
//
__int64 __fastcall sub_144BEA0(__int64 a1, __int64 *a2)
{
  __int64 v3; // r8
  __int64 v4; // rax
  __int64 *v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rsi
  __int64 *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rdx
  unsigned __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rcx
  char v17; // si
  __int64 v18; // rax
  __int64 v19; // rcx
  unsigned __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rax
  char v25; // si
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rdi
  __int64 v29; // r8
  __int64 v30; // rbx
  __int64 v31; // r12
  __int64 v32; // rax
  __int64 v33; // rdi
  int v34; // eax
  __int64 v35; // rsi
  __int64 v36; // rdi
  __int64 v37; // r15
  __int64 *v38; // rax
  char v39; // dl
  __int64 v40; // rax
  char v41; // r8
  char v42; // si
  __int64 *v44; // rsi
  __int64 *v45; // rdi
  __m128i *v46; // rax
  __m128i si128; // xmm0
  __int64 *v48; // rdi
  __int64 *v49; // rdx
  __int64 v50; // [rsp+10h] [rbp-260h]
  __int64 v51[3]; // [rsp+20h] [rbp-250h] BYREF
  char v52; // [rsp+38h] [rbp-238h]
  __int64 v53; // [rsp+40h] [rbp-230h] BYREF
  __int64 *v54; // [rsp+48h] [rbp-228h]
  __int64 *v55; // [rsp+50h] [rbp-220h]
  __int64 v56; // [rsp+58h] [rbp-218h]
  int v57; // [rsp+60h] [rbp-210h]
  _QWORD v58[8]; // [rsp+68h] [rbp-208h] BYREF
  __int64 v59; // [rsp+A8h] [rbp-1C8h] BYREF
  __int64 v60; // [rsp+B0h] [rbp-1C0h]
  unsigned __int64 v61; // [rsp+B8h] [rbp-1B8h]
  _QWORD v62[16]; // [rsp+C0h] [rbp-1B0h] BYREF
  _QWORD v63[2]; // [rsp+140h] [rbp-130h] BYREF
  unsigned __int64 v64; // [rsp+150h] [rbp-120h]
  char v65; // [rsp+158h] [rbp-118h]
  char v66[64]; // [rsp+168h] [rbp-108h] BYREF
  __int64 v67; // [rsp+1A8h] [rbp-C8h]
  __int64 v68; // [rsp+1B0h] [rbp-C0h]
  unsigned __int64 v69; // [rsp+1B8h] [rbp-B8h]
  char v70[8]; // [rsp+1C0h] [rbp-B0h] BYREF
  __int64 v71; // [rsp+1C8h] [rbp-A8h]
  unsigned __int64 v72; // [rsp+1D0h] [rbp-A0h]
  char v73[64]; // [rsp+1E8h] [rbp-88h] BYREF
  __int64 v74; // [rsp+228h] [rbp-48h]
  __int64 v75; // [rsp+230h] [rbp-40h]
  __int64 v76; // [rsp+238h] [rbp-38h]

  sub_16E7EE0(*(_QWORD *)(a1 + 192), *(const char **)(a1 + 160), *(_QWORD *)(a1 + 168));
  memset(v62, 0, sizeof(v62));
  v3 = a2[4];
  v62[1] = &v62[5];
  v62[2] = &v62[5];
  v4 = *a2;
  v54 = v58;
  v58[0] = v4 & 0xFFFFFFFFFFFFFFF8LL;
  v63[0] = v4 & 0xFFFFFFFFFFFFFFF8LL;
  v55 = v58;
  v50 = v3;
  LODWORD(v62[3]) = 8;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v56 = 0x100000008LL;
  v57 = 0;
  v53 = 1;
  v65 = 0;
  sub_144A690(&v59, (__int64)v63);
  v5 = v58;
  v48 = &v54[HIDWORD(v56)];
  if ( v54 == v48 )
  {
LABEL_97:
    ++HIDWORD(v56);
    *v48 = v50;
    ++v53;
  }
  else
  {
    v49 = 0;
    while ( v50 != *v5 )
    {
      if ( *v5 == -2 )
        v49 = v5;
      if ( v48 == ++v5 )
      {
        if ( !v49 )
          goto LABEL_97;
        *v49 = v50;
        --v57;
        ++v53;
        break;
      }
    }
  }
  sub_16CCEE0(v63, v66, 8, &v53);
  v6 = v59;
  v59 = 0;
  v67 = v6;
  v7 = v60;
  v60 = 0;
  v68 = v7;
  v8 = v61;
  v61 = 0;
  v69 = v8;
  sub_16CCEE0(v70, v73, 8, v62);
  v9 = v62[13];
  memset(&v62[13], 0, 24);
  v74 = v9;
  v75 = v62[14];
  v76 = v62[15];
  if ( v59 )
    j_j___libc_free_0(v59, v61 - v59);
  if ( v55 != v54 )
    _libc_free((unsigned __int64)v55);
  if ( v62[13] )
    j_j___libc_free_0(v62[13], v62[15] - v62[13]);
  if ( v62[2] != v62[1] )
    _libc_free(v62[2]);
  v10 = v58;
  v11 = &v53;
  sub_16CCCB0(&v53, v58, v63);
  v12 = v68;
  v13 = v67;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v14 = v68 - v67;
  if ( v68 == v67 )
  {
    v15 = 0;
  }
  else
  {
    if ( v14 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_98;
    v15 = sub_22077B0(v68 - v67);
    v12 = v68;
    v13 = v67;
  }
  v59 = v15;
  v60 = v15;
  v61 = v15 + v14;
  if ( v12 == v13 )
  {
    v16 = v15;
  }
  else
  {
    v16 = v15 + v12 - v13;
    do
    {
      if ( v15 )
      {
        *(_QWORD *)v15 = *(_QWORD *)v13;
        v17 = *(_BYTE *)(v13 + 24);
        *(_BYTE *)(v15 + 24) = v17;
        if ( v17 )
          *(__m128i *)(v15 + 8) = _mm_loadu_si128((const __m128i *)(v13 + 8));
      }
      v15 += 32;
      v13 += 32;
    }
    while ( v16 != v15 );
  }
  v10 = &v62[5];
  v11 = v62;
  v60 = v16;
  sub_16CCCB0(v62, &v62[5], v70);
  v18 = v75;
  v19 = v74;
  memset(&v62[13], 0, 24);
  v20 = v75 - v74;
  if ( v75 != v74 )
  {
    if ( v20 <= 0x7FFFFFFFFFFFFFE0LL )
    {
      v21 = sub_22077B0(v75 - v74);
      v19 = v74;
      v22 = v21;
      v18 = v75;
      goto LABEL_22;
    }
LABEL_98:
    sub_4261EA(v11, v10, v13);
  }
  v22 = 0;
LABEL_22:
  v62[13] = v22;
  v62[14] = v22;
  v62[15] = v22 + v20;
  if ( v18 == v19 )
  {
    v24 = v22;
  }
  else
  {
    v23 = v22;
    v24 = v22 + v18 - v19;
    do
    {
      if ( v23 )
      {
        *(_QWORD *)v23 = *(_QWORD *)v19;
        v25 = *(_BYTE *)(v19 + 24);
        *(_BYTE *)(v23 + 24) = v25;
        if ( v25 )
          *(__m128i *)(v23 + 8) = _mm_loadu_si128((const __m128i *)(v19 + 8));
      }
      v23 += 32;
      v19 += 32;
    }
    while ( v24 != v23 );
  }
  v26 = v60;
  v27 = v59;
  v62[14] = v24;
  if ( v60 - v59 == v24 - v22 )
    goto LABEL_41;
  do
  {
LABEL_29:
    v28 = *(_QWORD *)(v26 - 32);
    v29 = *(_QWORD *)(a1 + 192);
    if ( v28 )
    {
      sub_155C2B0(v28, *(_QWORD *)(a1 + 192), 0);
    }
    else
    {
      v46 = *(__m128i **)(v29 + 24);
      if ( *(_QWORD *)(v29 + 16) - (_QWORD)v46 <= 0x14u )
      {
        sub_16E7EE0(*(_QWORD *)(a1 + 192), "Printing <null> Block", 21);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_4289F10);
        v46[1].m128i_i32[0] = 1668246594;
        v46[1].m128i_i8[4] = 107;
        *v46 = si128;
        *(_QWORD *)(v29 + 24) += 21LL;
      }
    }
    v30 = v60;
    do
    {
      v31 = *(_QWORD *)(v30 - 32);
      if ( !*(_BYTE *)(v30 - 8) )
      {
        v32 = sub_157EBA0(*(_QWORD *)(v30 - 32));
        *(_BYTE *)(v30 - 8) = 1;
        *(_QWORD *)(v30 - 24) = v32;
        *(_DWORD *)(v30 - 16) = 0;
      }
      while ( 1 )
      {
        v33 = sub_157EBA0(v31);
        v34 = 0;
        if ( v33 )
          v34 = sub_15F4D60(v33);
        v35 = *(unsigned int *)(v30 - 16);
        if ( (_DWORD)v35 == v34 )
          break;
        v36 = *(_QWORD *)(v30 - 24);
        *(_DWORD *)(v30 - 16) = v35 + 1;
        v37 = sub_15F4DF0(v36, v35);
        v38 = v54;
        if ( v55 != v54 )
          goto LABEL_38;
        v44 = &v54[HIDWORD(v56)];
        if ( v54 == v44 )
        {
LABEL_74:
          if ( HIDWORD(v56) < (unsigned int)v56 )
          {
            ++HIDWORD(v56);
            *v44 = v37;
            ++v53;
LABEL_39:
            v51[0] = v37;
            v52 = 0;
            sub_144A690(&v59, (__int64)v51);
            v26 = v60;
            v27 = v59;
            goto LABEL_40;
          }
LABEL_38:
          sub_16CCBA0(&v53, v37);
          if ( v39 )
            goto LABEL_39;
        }
        else
        {
          v45 = 0;
          while ( v37 != *v38 )
          {
            if ( *v38 == -2 )
            {
              v45 = v38;
              if ( v44 == v38 + 1 )
                goto LABEL_71;
              ++v38;
            }
            else if ( v44 == ++v38 )
            {
              if ( !v45 )
                goto LABEL_74;
LABEL_71:
              *v45 = v37;
              --v57;
              ++v53;
              goto LABEL_39;
            }
          }
        }
      }
      v60 -= 32;
      v26 = v59;
      v30 = v60;
    }
    while ( v60 != v59 );
    v27 = v59;
LABEL_40:
    v22 = v62[13];
  }
  while ( v26 - v27 != v62[14] - v62[13] );
LABEL_41:
  if ( v26 != v27 )
  {
    v40 = v22;
    while ( *(_QWORD *)v27 == *(_QWORD *)v40 )
    {
      v41 = *(_BYTE *)(v27 + 24);
      v42 = *(_BYTE *)(v40 + 24);
      if ( v41 && v42 )
      {
        if ( *(_DWORD *)(v27 + 16) != *(_DWORD *)(v40 + 16) )
          goto LABEL_29;
        v27 += 32;
        v40 += 32;
        if ( v26 == v27 )
          goto LABEL_48;
      }
      else
      {
        if ( v41 != v42 )
          goto LABEL_29;
        v27 += 32;
        v40 += 32;
        if ( v26 == v27 )
          goto LABEL_48;
      }
    }
    goto LABEL_29;
  }
LABEL_48:
  if ( v22 )
    j_j___libc_free_0(v22, v62[15] - v22);
  if ( v62[2] != v62[1] )
    _libc_free(v62[2]);
  if ( v59 )
    j_j___libc_free_0(v59, v61 - v59);
  if ( v55 != v54 )
    _libc_free((unsigned __int64)v55);
  if ( v74 )
    j_j___libc_free_0(v74, v76 - v74);
  if ( v72 != v71 )
    _libc_free(v72);
  if ( v67 )
    j_j___libc_free_0(v67, v69 - v67);
  if ( v64 != v63[1] )
    _libc_free(v64);
  return 0;
}
