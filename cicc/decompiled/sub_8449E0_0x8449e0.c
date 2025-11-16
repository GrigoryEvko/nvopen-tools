// Function: sub_8449E0
// Address: 0x8449e0
//
__int64 __fastcall sub_8449E0(_QWORD *a1, __m128i *a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  char v8; // al
  __int64 v9; // rsi
  char v10; // al
  __int64 v11; // r10
  char v12; // di
  _BOOL4 v13; // r14d
  _BOOL8 v14; // rdx
  unsigned int v15; // r13d
  int v17; // r8d
  __int64 v18; // r10
  char v19; // dl
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // r10
  __int64 i; // rax
  char v24; // al
  _BOOL4 v25; // eax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // r10
  _QWORD *v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rcx
  __int64 v34; // r8
  char v35; // al
  char v36; // al
  char v37; // cl
  __int64 v38; // rax
  char j; // dl
  char v40; // al
  unsigned int v41; // eax
  __int64 v42; // rbx
  __int64 v43; // [rsp+8h] [rbp-1B8h]
  _BYTE *v44; // [rsp+10h] [rbp-1B0h]
  __int64 v45; // [rsp+10h] [rbp-1B0h]
  __int64 v46; // [rsp+10h] [rbp-1B0h]
  __int64 v47; // [rsp+10h] [rbp-1B0h]
  __int64 v48; // [rsp+18h] [rbp-1A8h]
  __int64 v49; // [rsp+18h] [rbp-1A8h]
  __int64 v50; // [rsp+18h] [rbp-1A8h]
  int v51; // [rsp+18h] [rbp-1A8h]
  __int64 v52; // [rsp+18h] [rbp-1A8h]
  __int64 v53; // [rsp+18h] [rbp-1A8h]
  __int64 v54; // [rsp+18h] [rbp-1A8h]
  int v55; // [rsp+18h] [rbp-1A8h]
  int v56; // [rsp+18h] [rbp-1A8h]
  __int64 v57; // [rsp+20h] [rbp-1A0h] BYREF
  __int64 v58; // [rsp+28h] [rbp-198h] BYREF
  _OWORD v59[4]; // [rsp+30h] [rbp-190h] BYREF
  _OWORD v60[5]; // [rsp+70h] [rbp-150h] BYREF
  __m128i v61; // [rsp+C0h] [rbp-100h]
  __m128i v62; // [rsp+D0h] [rbp-F0h]
  __m128i v63; // [rsp+E0h] [rbp-E0h]
  __m128i v64; // [rsp+F0h] [rbp-D0h]
  __m128i v65; // [rsp+100h] [rbp-C0h]
  __m128i v66; // [rsp+110h] [rbp-B0h]
  __m128i v67; // [rsp+120h] [rbp-A0h]
  __m128i v68; // [rsp+130h] [rbp-90h]
  __m128i v69; // [rsp+140h] [rbp-80h]
  __m128i v70; // [rsp+150h] [rbp-70h]
  __m128i v71; // [rsp+160h] [rbp-60h]
  __m128i v72; // [rsp+170h] [rbp-50h]
  __m128i v73; // [rsp+180h] [rbp-40h]

  v8 = *((_BYTE *)a1 + 16);
  v59[0] = _mm_loadu_si128((const __m128i *)a1);
  v59[1] = _mm_loadu_si128((const __m128i *)a1 + 1);
  v59[2] = _mm_loadu_si128((const __m128i *)a1 + 2);
  v59[3] = _mm_loadu_si128((const __m128i *)a1 + 3);
  v60[0] = _mm_loadu_si128((const __m128i *)a1 + 4);
  v60[1] = _mm_loadu_si128((const __m128i *)a1 + 5);
  v60[2] = _mm_loadu_si128((const __m128i *)a1 + 6);
  v60[3] = _mm_loadu_si128((const __m128i *)a1 + 7);
  v60[4] = _mm_loadu_si128((const __m128i *)a1 + 8);
  if ( v8 == 2 )
  {
    v61 = _mm_loadu_si128((const __m128i *)a1 + 9);
    v62 = _mm_loadu_si128((const __m128i *)a1 + 10);
    v63 = _mm_loadu_si128((const __m128i *)a1 + 11);
    v64 = _mm_loadu_si128((const __m128i *)a1 + 12);
    v65 = _mm_loadu_si128((const __m128i *)a1 + 13);
    v66 = _mm_loadu_si128((const __m128i *)a1 + 14);
    v67 = _mm_loadu_si128((const __m128i *)a1 + 15);
    v68 = _mm_loadu_si128((const __m128i *)a1 + 16);
    v69 = _mm_loadu_si128((const __m128i *)a1 + 17);
    v70 = _mm_loadu_si128((const __m128i *)a1 + 18);
    v71 = _mm_loadu_si128((const __m128i *)a1 + 19);
    v72 = _mm_loadu_si128((const __m128i *)a1 + 20);
    v73 = _mm_loadu_si128((const __m128i *)a1 + 21);
  }
  else if ( v8 == 5 || v8 == 1 )
  {
    v61.m128i_i64[0] = a1[18];
  }
  v9 = *(unsigned __int8 *)(a3 + 17);
  v10 = *(_BYTE *)(a3 + 16);
  v11 = *(_QWORD *)a3;
  v12 = v10 & 1;
  v13 = (*(_BYTE *)(a3 + 17) & 2) != 0;
  v14 = (*(_BYTE *)(a3 + 17) & 2) == 0;
  v15 = v14;
  if ( (v10 & 4) != 0 )
  {
    if ( v12 )
    {
      sub_8424A0((__m128i *)a1, a2, v14, (__int64)a4, a5);
      return sub_6E4F10((__int64)a1, (__int64)v59, v15, 0);
    }
    v9 &= 1u;
    if ( (_DWORD)v9 )
      goto LABEL_7;
LABEL_20:
    if ( (v10 & 2) != 0 && a2 )
    {
      if ( (__m128i *)*a1 == a2
        || (v9 = (__int64)a2, v44 = a4, v49 = v11, (unsigned int)sub_8D97D0(*a1, a2, 32, a4, a5)) )
      {
        *(_BYTE *)(a3 + 16) |= 0x10u;
        v32 = (__int64)a2;
        sub_831640((__m128i *)a1, a2, *(_QWORD *)(a3 + 24), (__int64)a4, a5);
        if ( (*(_BYTE *)(a3 + 16) & 4) != 0 )
          return sub_6E4F10((__int64)a1, (__int64)v59, v15, 0);
        goto LABEL_54;
      }
      v11 = v49;
      a4 = v44;
    }
    if ( v11 )
    {
      if ( *(_BYTE *)(v11 + 174) != 3 )
      {
        v50 = v11;
        sub_8441D0((__m128i *)a1, v11, (*(_BYTE *)(a3 + 17) & 4) != 0, a4, &v57, &v58);
        v17 = v58;
        v18 = v50;
        if ( (_DWORD)v58 )
          return sub_6E4F10((__int64)a1, (__int64)v59, v15, 0);
        goto LABEL_18;
      }
      goto LABEL_37;
    }
    a5 = 0;
    goto LABEL_68;
  }
  if ( v12 )
  {
    v51 = a5;
    sub_8424A0((__m128i *)a1, a2, v14, (__int64)a4, a5);
    if ( v51 && *((_BYTE *)a1 + 16) )
    {
      v19 = *(_BYTE *)(*a1 + 140LL);
      if ( v19 == 12 )
      {
        v20 = *a1;
        do
        {
          v20 = *(_QWORD *)(v20 + 160);
          v19 = *(_BYTE *)(v20 + 140);
        }
        while ( v19 == 12 );
      }
      if ( v19 )
      {
        sub_6E61E0(*a1, (__int64)a1 + 68, 0);
        sub_8283A0((__int64)a1, 0, 0, v13);
      }
    }
    return sub_6E4F10((__int64)a1, (__int64)v59, v15, 0);
  }
  if ( (v9 & 1) != 0 )
  {
LABEL_7:
    if ( !a2 )
      a2 = (__m128i *)dword_4D03B80;
    sub_6F4200((__m128i *)a1, (__int64)a2, 0, v14);
    sub_6E41D0((__int64)a1, 0, 0, (_QWORD *)((char *)a1 + 68), (_QWORD *)((char *)a1 + 68), (__int64)a2);
    return sub_6E4F10((__int64)a1, (__int64)v59, v15, 0);
  }
  if ( !(_DWORD)a5 )
    goto LABEL_20;
  if ( !v11 )
  {
    if ( *((_BYTE *)a1 + 17) != 1 )
    {
      if ( (v10 & 0x10) == 0 )
        goto LABEL_64;
LABEL_71:
      v9 = (__int64)a2;
      v56 = a5;
      sub_831640((__m128i *)a1, a2, *(_QWORD *)(a3 + 24), (__int64)a4, a5);
      v36 = *(_BYTE *)(a3 + 16);
      LODWORD(a5) = v56;
LABEL_69:
      if ( (v36 & 4) != 0 )
      {
LABEL_65:
        if ( (_DWORD)a5 )
          sub_8443E0((__m128i *)a1, (__int64)a2, 0);
        return sub_6E4F10((__int64)a1, (__int64)v59, v15, 0);
      }
LABEL_64:
      v55 = a5;
      sub_6FA3A0((__m128i *)a1, v9);
      LODWORD(a5) = v55;
      goto LABEL_65;
    }
    *(_BYTE *)(a3 + 16) = v10 | 4;
LABEL_68:
    v36 = *(_BYTE *)(a3 + 16);
    if ( (v36 & 0x10) == 0 )
      goto LABEL_69;
    goto LABEL_71;
  }
  if ( *(_BYTE *)(v11 + 174) != 3 )
  {
    v48 = *(_QWORD *)a3;
    sub_8441D0((__m128i *)a1, v11, (v9 & 4) != 0, a4, &v57, &v58);
    v17 = v58;
    v18 = v48;
LABEL_18:
    sub_8284D0(
      v18,
      v57,
      (__int64)a2,
      0,
      v17,
      v13,
      (*(_BYTE *)(a3 + 17) & 4) != 0,
      (__int64 *)((char *)v60 + 4),
      (__int64)a1);
    return sub_6E4F10((__int64)a1, (__int64)v59, v15, 0);
  }
LABEL_37:
  v43 = v11;
  v45 = *(_QWORD *)(v11 + 152);
  sub_6E6130(*(_QWORD *)(a3 + 8), (_DWORD)a1 + 68, *a1, 1);
  v21 = v45;
  v22 = v43;
  for ( i = v45; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( *(_QWORD *)(*(_QWORD *)(i + 168) + 40LL) )
  {
    if ( HIDWORD(qword_4D0495C) )
    {
      if ( (*(_BYTE *)(*a1 + 140LL) & 0xFB) == 8 )
      {
        v24 = sub_8D4C10(*a1, dword_4F077C4 != 2);
        v22 = v43;
        v21 = v45;
        if ( (v24 & 1) != 0 && (*(_BYTE *)(*(_QWORD *)(v45 + 168) + 18LL) & 1) == 0 )
        {
          v25 = sub_6E53E0(5, 0x20Cu, (_DWORD *)a1 + 17);
          v22 = v43;
          v21 = v45;
          if ( v25 )
          {
            sub_684B30(0x20Cu, (_DWORD *)a1 + 17);
            v21 = v45;
            v22 = v43;
          }
        }
      }
    }
    if ( *((_WORD *)a1 + 8) == 514 && *((_BYTE *)a1 + 317) != 12 && (*(_BYTE *)(*(_QWORD *)(v21 + 168) + 18LL) & 1) == 0 )
    {
      v47 = v21;
      v54 = v22;
      sub_844770((__m128i *)a1, 0);
      v21 = v47;
      v22 = v54;
    }
    v46 = v22;
    v52 = v21;
    sub_831410(v21, (__int64)a1);
    sub_8316D0((__m128i *)a1, v52);
    v30 = v46;
  }
  else
  {
    sub_843D70((__m128i *)a1, **(_QWORD **)(v45 + 168), 0, 0xA7u);
    v30 = v43;
  }
  v53 = v30;
  v57 = sub_6F6F40((const __m128i *)a1, 0, v26, v27, v28, v29);
  v31 = sub_7312D0(v53);
  *(_QWORD *)((char *)v31 + 28) = *(_QWORD *)((char *)a1 + 68);
  v31[2] = v57;
  v32 = *(_QWORD *)(v53 + 152);
  sub_701D00(
    v31,
    v32,
    (*(_BYTE *)(v53 + 192) & 2) != 0,
    0,
    (*((_BYTE *)a1 + 18) & 2) != 0,
    v15,
    1,
    0,
    0,
    0,
    0,
    (__int64 *)&dword_4F077C8,
    (_DWORD *)v60 + 1,
    &dword_4F077C8,
    (__m128i *)a1,
    0,
    &v58);
  if ( a2 )
  {
    if ( !(unsigned int)sub_8D3A70(*a1) && !(unsigned int)sub_8D3A70(a2) )
    {
      if ( (*(_BYTE *)(a3 + 16) & 4) == 0 || (*(_BYTE *)(a3 + 36) & 0x20) != 0 )
        sub_6F69D0(a1, 0);
      v37 = *((_BYTE *)a1 + 16);
      if ( v37 )
      {
        v38 = *a1;
        for ( j = *(_BYTE *)(*a1 + 140LL); j == 12; j = *(_BYTE *)(v38 + 140) )
          v38 = *(_QWORD *)(v38 + 160);
        if ( j )
        {
          v40 = *((_BYTE *)a1 + 17);
          if ( v40 == 2 )
          {
            v42 = 0;
            if ( v37 == 1 )
              v42 = a1[18];
            sub_6FC3F0((__int64)a2, (__m128i *)a1, v15);
            if ( v13 && v58 && *((_BYTE *)a1 + 16) == 1 && a1[18] != v42 )
              *(_BYTE *)(v58 + 27) |= 2u;
          }
          else if ( v40 == 1 )
          {
            sub_6F7690((const __m128i *)a1, (__int64)a2);
          }
        }
      }
      return sub_6E4F10((__int64)a1, (__int64)v59, v15, 0);
    }
    if ( (*(_BYTE *)(*a1 + 140LL) & 0xFB) == 8 )
    {
      v41 = sub_8D4C10(*a1, dword_4F077C4 != 2);
      v32 = v41;
      if ( v41 )
      {
        if ( !dword_4F077BC || (_DWORD)qword_4F077B4 || qword_4F077A8 > 0x9EFBu )
          a2 = sub_73C570(a2, v41);
      }
    }
    v35 = *(_BYTE *)(a3 + 16);
    if ( (v35 & 0x10) != 0 )
    {
      v32 = (__int64)a2;
      sub_831640((__m128i *)a1, a2, *(_QWORD *)(a3 + 24), v33, v34);
      v35 = *(_BYTE *)(a3 + 16);
    }
    if ( (v35 & 4) == 0 )
LABEL_54:
      sub_6FA3A0((__m128i *)a1, v32);
  }
  else if ( (*(_BYTE *)(a3 + 16) & 4) == 0 )
  {
    sub_6F69D0(a1, 0);
  }
  return sub_6E4F10((__int64)a1, (__int64)v59, v15, 0);
}
