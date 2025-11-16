// Function: sub_6D6050
// Address: 0x6d6050
//
__int64 __fastcall sub_6D6050(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  unsigned int v6; // r15d
  __int64 v7; // rax
  char v8; // dl
  bool v9; // bl
  unsigned __int64 v10; // rsi
  __int64 v11; // r15
  bool v12; // bl
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdi
  __int64 v16; // r11
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  char *v22; // rsi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  char v34; // al
  __int64 v35; // rax
  char i; // dl
  char v37; // al
  __int64 v38; // rdi
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  int v42; // eax
  char v43; // al
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // [rsp-10h] [rbp-290h]
  __int64 v47; // [rsp+8h] [rbp-278h]
  __int64 v50; // [rsp+20h] [rbp-260h]
  bool v51; // [rsp+2Bh] [rbp-255h]
  unsigned int v52; // [rsp+2Ch] [rbp-254h]
  unsigned int v53; // [rsp+3Ch] [rbp-244h] BYREF
  __int64 v54; // [rsp+40h] [rbp-240h] BYREF
  __int64 v55; // [rsp+48h] [rbp-238h] BYREF
  _BYTE v56[18]; // [rsp+50h] [rbp-230h] BYREF
  char v57; // [rsp+62h] [rbp-21Eh]
  __m128i v58; // [rsp+F0h] [rbp-190h] BYREF
  __m128i v59; // [rsp+100h] [rbp-180h] BYREF
  __m128i v60; // [rsp+110h] [rbp-170h] BYREF
  __m128i v61; // [rsp+120h] [rbp-160h] BYREF
  __int64 v62; // [rsp+130h] [rbp-150h] BYREF
  __int64 v63; // [rsp+13Ch] [rbp-144h] BYREF
  __m128i v64; // [rsp+150h] [rbp-130h] BYREF
  __m128i v65; // [rsp+160h] [rbp-120h] BYREF
  __m128i v66; // [rsp+170h] [rbp-110h] BYREF
  __m128i v67; // [rsp+180h] [rbp-100h] BYREF
  __m128i v68; // [rsp+190h] [rbp-F0h] BYREF
  __m128i v69; // [rsp+1A0h] [rbp-E0h] BYREF
  __m128i v70; // [rsp+1B0h] [rbp-D0h] BYREF
  __m128i v71; // [rsp+1C0h] [rbp-C0h] BYREF
  __m128i v72; // [rsp+1D0h] [rbp-B0h] BYREF
  __m128i v73; // [rsp+1E0h] [rbp-A0h] BYREF
  __m128i v74; // [rsp+1F0h] [rbp-90h] BYREF
  __m128i v75; // [rsp+200h] [rbp-80h] BYREF
  __m128i v76; // [rsp+210h] [rbp-70h] BYREF
  __m128i v77; // [rsp+220h] [rbp-60h] BYREF
  __m128i v78; // [rsp+230h] [rbp-50h] BYREF
  __m128i v79[4]; // [rsp+240h] [rbp-40h] BYREF

  v4 = a1;
  v52 = unk_4D03B6C;
  v55 = *(_QWORD *)&dword_4F063F8;
  sub_6E1E00(2, v56, 0, 0);
  v57 |= 1u;
  sub_7296F0((unsigned int)dword_4F04C64, &v53);
  v6 = dword_4F04C3C;
  v7 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v8 = *(_BYTE *)(v7 + 7);
  v9 = (v8 & 0x10) != 0;
  if ( unk_4F07270 != unk_4F073B8 )
  {
    dword_4F04C3C = 1;
    *(_BYTE *)(v7 + 7) = v8 | 0x10;
  }
  v10 = 0;
  sub_69ED20((__int64)&v58, 0, 0, 1);
  if ( unk_4F07270 != unk_4F073B8 )
  {
    v10 = (unsigned __int64)qword_4F04C68;
    dword_4F04C3C = v6;
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7)
                                                             & 0xEF
                                                             | (16 * v9);
  }
  v11 = 0;
  v47 = 0;
  v12 = (v59.m128i_i8[2] & 8) != 0;
  v51 = v12 || (v59.m128i_i8[2] & 0x10) != 0;
  if ( v51 && v59.m128i_i8[0] == 1 )
  {
    v34 = *(_BYTE *)(v67.m128i_i64[0] + 24);
    if ( v34 == 3 )
    {
      v47 = *(_QWORD *)(v67.m128i_i64[0] + 56);
    }
    else if ( v34 == 20 )
    {
      v11 = *(_QWORD *)(v67.m128i_i64[0] + 56);
    }
  }
  sub_68AD60(&v58);
  if ( !a1 )
  {
    if ( v52 != unk_4D03B6C )
      v59.m128i_i8[4] |= 8u;
    goto LABEL_31;
  }
  v15 = sub_8D4940(a1);
  if ( (unsigned int)sub_8D3EA0(v15) || (unsigned int)sub_8D3F60(v15) )
  {
    v54 = 0;
    v16 = sub_6E2EF0(v15, v10, v13, v14);
    *(__m128i *)(v16 + 8) = _mm_loadu_si128(&v58);
    *(__m128i *)(v16 + 24) = _mm_loadu_si128(&v59);
    *(__m128i *)(v16 + 40) = _mm_loadu_si128(&v60);
    *(__m128i *)(v16 + 56) = _mm_loadu_si128(&v61);
    *(__m128i *)(v16 + 72) = _mm_loadu_si128((const __m128i *)&v62);
    *(__m128i *)(v16 + 88) = _mm_loadu_si128((const __m128i *)((char *)&v63 + 4));
    *(__m128i *)(v16 + 104) = _mm_loadu_si128(&v64);
    *(__m128i *)(v16 + 120) = _mm_loadu_si128(&v65);
    *(__m128i *)(v16 + 136) = _mm_loadu_si128(&v66);
    if ( v59.m128i_i8[0] == 2 )
    {
      *(__m128i *)(v16 + 152) = _mm_loadu_si128(&v67);
      *(__m128i *)(v16 + 168) = _mm_loadu_si128(&v68);
      *(__m128i *)(v16 + 184) = _mm_loadu_si128(&v69);
      *(__m128i *)(v16 + 200) = _mm_loadu_si128(&v70);
      *(__m128i *)(v16 + 216) = _mm_loadu_si128(&v71);
      *(__m128i *)(v16 + 232) = _mm_loadu_si128(&v72);
      *(__m128i *)(v16 + 248) = _mm_loadu_si128(&v73);
      *(__m128i *)(v16 + 264) = _mm_loadu_si128(&v74);
      *(__m128i *)(v16 + 280) = _mm_loadu_si128(&v75);
      *(__m128i *)(v16 + 296) = _mm_loadu_si128(&v76);
      *(__m128i *)(v16 + 312) = _mm_loadu_si128(&v77);
      *(__m128i *)(v16 + 328) = _mm_loadu_si128(&v78);
      *(__m128i *)(v16 + 344) = _mm_loadu_si128(v79);
    }
    else if ( v59.m128i_i8[0] == 5 || v59.m128i_i8[0] == 1 )
    {
      *(_QWORD *)(v16 + 152) = v67.m128i_i64[0];
    }
    v17 = a3;
    v10 = 0;
    v50 = v16;
    sub_696F90(v4, 0, v16, &v54, &v55, v17, a4);
    v4 = v54;
    sub_6E1940(v50);
    v13 = v46;
    if ( unk_4D03B6C == v52 )
    {
LABEL_14:
      if ( v4 )
        goto LABEL_15;
LABEL_31:
      if ( dword_4F04C44 != -1
        || (v24 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v24 + 6) & 6) != 0)
        || *(_BYTE *)(v24 + 4) == 12 )
      {
        sub_6F5780(&v58, v10, v13);
      }
      else
      {
        sub_6F6C80(&v58);
      }
      v22 = (char *)a2;
      sub_6F4950(&v58, a2, v25, v26, v27, v28);
      goto LABEL_22;
    }
LABEL_13:
    v59.m128i_i8[4] |= 8u;
    goto LABEL_14;
  }
  v10 = v52;
  if ( v52 != unk_4D03B6C )
    goto LABEL_13;
LABEL_15:
  if ( (unsigned int)sub_8D3EA0(v4) )
  {
    v22 = (char *)a2;
    sub_695E70((__int64)&v58, a2, v18, v19, v20, v21);
  }
  else if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 && (unsigned int)sub_8D3D40(v4) )
  {
    sub_6F5780(&v58, v10, v29);
    sub_6F4200(&v58, v4, 0, 1);
    v22 = (char *)a2;
    sub_695E70((__int64)&v58, a2, v30, v31, v32, v33);
    if ( *(_BYTE *)(a2 + 173) == 12 )
      *(_BYTE *)(a2 + 177) |= 8u;
  }
  else
  {
    v22 = (char *)v4;
    sub_695B00(&v58, v4, a2);
  }
  while ( *(_BYTE *)(v4 + 140) == 12 )
    v4 = *(_QWORD *)(v4 + 160);
  if ( !(unsigned int)sub_8DBE70(v4) && *(_BYTE *)(a2 + 173) )
  {
    v35 = *(_QWORD *)(a2 + 128);
    for ( i = *(_BYTE *)(v35 + 140); i == 12; i = *(_BYTE *)(v35 + 140) )
      v35 = *(_QWORD *)(v35 + 160);
    if ( i )
    {
      v37 = *(_BYTE *)(v4 + 140);
      if ( v37 != 6 && v37 != 13 )
      {
        if ( (unsigned __int8)(v37 - 9) <= 2u && !(unsigned int)sub_730B80(a2) )
        {
          v22 = (char *)&v62 + 4;
          sub_685360(0xA55u, (_DWORD *)&v62 + 1, v58.m128i_i64[0]);
        }
        goto LABEL_22;
      }
      if ( unk_4D0480C )
      {
        if ( v59.m128i_i8[0] != 2 )
        {
LABEL_57:
          v38 = a2;
          sub_730990(a2);
LABEL_58:
          if ( (unsigned int)sub_6E5430(v38, v22, v39, v40, v41) )
          {
            v22 = (char *)&v62 + 4;
            sub_685360(0xA55u, (_DWORD *)&v62 + 1, v58.m128i_i64[0]);
          }
          sub_72C970(a2);
          goto LABEL_22;
        }
        goto LABEL_75;
      }
      if ( (unsigned int)sub_8D32E0(v4) )
      {
        v38 = a2;
        v42 = sub_730990(a2);
        if ( !v12 )
          goto LABEL_58;
LABEL_63:
        if ( v42 )
          goto LABEL_22;
        goto LABEL_58;
      }
      if ( v59.m128i_i8[0] != 2 )
        goto LABEL_57;
      if ( *(_BYTE *)(a2 + 173) == 6 )
      {
        v43 = *(_BYTE *)(a2 + 176);
        if ( v43 == 1 )
        {
          v45 = *(_QWORD *)(a2 + 184);
          if ( v45 )
          {
            if ( !dword_4F077BC || (_DWORD)qword_4F077B4 )
            {
              if ( !v51 || v47 && v47 != v45 )
                goto LABEL_57;
              goto LABEL_75;
            }
            goto LABEL_72;
          }
        }
        else if ( !v43 )
        {
          v44 = *(_QWORD *)(a2 + 184);
          if ( v44 )
          {
            if ( !dword_4F077BC || (_DWORD)qword_4F077B4 )
            {
              if ( !v51 || v11 && v11 != v44 )
                goto LABEL_57;
              goto LABEL_75;
            }
LABEL_72:
            if ( (qword_4F077A8 > 0x76BFu || v51) && !v51 )
              goto LABEL_57;
          }
        }
      }
LABEL_75:
      v38 = a2;
      v42 = sub_730990(a2);
      goto LABEL_63;
    }
  }
LABEL_22:
  sub_6959C0(a2, (__int64)v22);
  sub_729730(v53);
  if ( *(_QWORD *)(a2 + 144) && !*(_BYTE *)(qword_4D03C50 + 16LL) )
    *(_QWORD *)(a2 + 144) = 0;
  sub_6E2AC0(a2);
  sub_6E2B30(a2, v22);
  *(_QWORD *)&dword_4F061D8 = v63;
  return sub_729730(v53);
}
