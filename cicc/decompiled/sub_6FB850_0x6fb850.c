// Function: sub_6FB850
// Address: 0x6fb850
//
__int64 __fastcall sub_6FB850(
        __int64 a1,
        __m128i *a2,
        _DWORD *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7,
        int a8)
{
  _DWORD *v8; // r14
  unsigned int v9; // r13d
  __int64 v11; // r15
  __int8 v12; // al
  char v13; // dl
  __int64 v14; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int8 v20; // al
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // r13
  bool v28; // zf
  __int64 v29; // rax
  _DWORD *v30; // rsi
  __int64 v31; // rcx
  int v32; // r10d
  __int64 v33; // rcx
  __int64 v34; // rax
  int v35; // eax
  __int64 v36; // rax
  __int64 v37; // rax
  unsigned int v38; // [rsp+8h] [rbp-1B8h]
  __int16 v39; // [rsp+8h] [rbp-1B8h]
  unsigned int v40; // [rsp+8h] [rbp-1B8h]
  unsigned int v41; // [rsp+Ch] [rbp-1B4h]
  unsigned int v42; // [rsp+1Ch] [rbp-1A4h] BYREF
  __int64 v43; // [rsp+20h] [rbp-1A0h] BYREF
  __int64 *v44; // [rsp+28h] [rbp-198h] BYREF
  _OWORD v45[4]; // [rsp+30h] [rbp-190h] BYREF
  _OWORD v46[5]; // [rsp+70h] [rbp-150h] BYREF
  __m128i v47; // [rsp+C0h] [rbp-100h]
  __m128i v48; // [rsp+D0h] [rbp-F0h]
  __m128i v49; // [rsp+E0h] [rbp-E0h]
  __m128i v50; // [rsp+F0h] [rbp-D0h]
  __m128i v51; // [rsp+100h] [rbp-C0h]
  __m128i v52; // [rsp+110h] [rbp-B0h]
  __m128i v53; // [rsp+120h] [rbp-A0h]
  __m128i v54; // [rsp+130h] [rbp-90h]
  __m128i v55; // [rsp+140h] [rbp-80h]
  __m128i v56; // [rsp+150h] [rbp-70h]
  __m128i v57; // [rsp+160h] [rbp-60h]
  __m128i v58; // [rsp+170h] [rbp-50h]
  __m128i v59; // [rsp+180h] [rbp-40h]

  v8 = a3;
  v9 = a4;
  v38 = a5;
  v41 = a6;
  v43 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  if ( !(unsigned int)sub_6E6010() )
    v9 = 0;
  v11 = sub_73D720(a1);
  v45[0] = _mm_loadu_si128(a2);
  v12 = a2[1].m128i_i8[0];
  v45[1] = _mm_loadu_si128(a2 + 1);
  v45[2] = _mm_loadu_si128(a2 + 2);
  v45[3] = _mm_loadu_si128(a2 + 3);
  v46[0] = _mm_loadu_si128(a2 + 4);
  v46[1] = _mm_loadu_si128(a2 + 5);
  v46[2] = _mm_loadu_si128(a2 + 6);
  v46[3] = _mm_loadu_si128(a2 + 7);
  v46[4] = _mm_loadu_si128(a2 + 8);
  if ( v12 == 2 )
  {
    v47 = _mm_loadu_si128(a2 + 9);
    v48 = _mm_loadu_si128(a2 + 10);
    v49 = _mm_loadu_si128(a2 + 11);
    v50 = _mm_loadu_si128(a2 + 12);
    v51 = _mm_loadu_si128(a2 + 13);
    v52 = _mm_loadu_si128(a2 + 14);
    v53 = _mm_loadu_si128(a2 + 15);
    v54 = _mm_loadu_si128(a2 + 16);
    v55 = _mm_loadu_si128(a2 + 17);
    v56 = _mm_loadu_si128(a2 + 18);
    v57 = _mm_loadu_si128(a2 + 19);
    v58 = _mm_loadu_si128(a2 + 20);
    v59 = _mm_loadu_si128(a2 + 21);
  }
  else if ( v12 == 5 || v12 == 1 )
  {
    v47.m128i_i64[0] = a2[9].m128i_i64[0];
  }
  v13 = *(_BYTE *)(v11 + 140);
  if ( !v8 )
    v8 = (_DWORD *)v46 + 1;
  if ( v13 == 12 )
  {
    v14 = v11;
    do
    {
      v14 = *(_QWORD *)(v14 + 160);
      v13 = *(_BYTE *)(v14 + 140);
    }
    while ( v13 == 12 );
  }
  if ( !v13 )
  {
    sub_6E6840((__int64)a2);
    goto LABEL_13;
  }
  if ( (unsigned int)sub_8D2600(v11) )
  {
    if ( !(unsigned int)sub_8D2600(a2->m128i_i64[0]) )
      sub_6F7220(a2, v11);
    goto LABEL_13;
  }
  v20 = a2[1].m128i_i8[0];
  if ( v20 == 2 )
  {
    sub_72A510(&a2[9], v43);
    if ( !a8 || !dword_4D0488C )
      goto LABEL_41;
    v23 = (unsigned int)qword_4F077B4;
    if ( dword_4F077BC )
    {
      if ( (_DWORD)qword_4F077B4 )
      {
LABEL_33:
        v23 = (__int64)&qword_4F077A0;
        v24 = qword_4D03C50;
        v25 = *(unsigned __int8 *)(qword_4D03C50 + 16LL);
        if ( !qword_4F077A0 || (_BYTE)v25 != 4 )
          goto LABEL_35;
LABEL_41:
        sub_6E5170(v43, v11, v41, v9, v38, a8, 0, (__int64)&v42, (__int64)v8);
        v26 = v42;
        if ( v42 )
        {
          v24 = qword_4D03C50;
          v25 = *(unsigned __int8 *)(qword_4D03C50 + 16LL);
          goto LABEL_36;
        }
        v27 = v43;
        v28 = *(_QWORD *)(v43 + 136) == 0;
        *(_BYTE *)(v43 + 168) = *(_BYTE *)(v43 + 168) & 0xBF | ((a7 & 1 | ((*(_BYTE *)(v43 + 168) & 0x40) != 0)) << 6);
        if ( v28 )
        {
          v40 = v26;
          v35 = sub_8DAAE0(a2->m128i_i64[0], v11);
          v27 = v43;
          if ( v35 )
          {
            v26 = v40;
          }
          else if ( dword_4F07590
                 || (v26 = 1, dword_4F04C44 == -1)
                 && (v36 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v36 + 6) & 6) == 0)
                 && *(_BYTE *)(v36 + 4) != 12 )
          {
            v37 = sub_8E3250(a2->m128i_i64[0]);
            v26 = 0;
            *(_QWORD *)(v27 + 136) = v37;
            v27 = v43;
          }
        }
        v29 = qword_4D03C50;
        if ( !*(_BYTE *)(qword_4D03C50 + 16LL) )
        {
LABEL_58:
          if ( (*(_BYTE *)(v29 + 21) & 0x40) != 0 && a2[18].m128i_i64[0] && !*(_QWORD *)(v27 + 144) )
          {
            sub_731DB0();
            v27 = v43;
          }
          sub_6E6A50(v27, (__int64)a2);
          sub_6E5070((__int64)a2, (__int64)v45);
          goto LABEL_13;
        }
        v30 = dword_4D03C08;
        v31 = a2[18].m128i_i64[0];
        v32 = dword_4D03C08[0];
        *(_QWORD *)(v27 + 144) = v31;
        if ( !v32 || (v26 &= 1u, (_DWORD)v26) )
        {
          v21 = v41;
          if ( !v41 )
            goto LABEL_51;
        }
        else
        {
          v22 = v41;
          if ( !v41 )
            goto LABEL_51;
          if ( (*(_BYTE *)(v29 + 19) & 4) != 0
            && !(*(_QWORD *)v29 | v31)
            && !a2[9].m128i_i64[1]
            && (unsigned int)sub_8D2DD0(v11) )
          {
            goto LABEL_75;
          }
        }
        v30 = (_DWORD *)v11;
        if ( (unsigned int)sub_8DAAE0(a2->m128i_i64[0], v11) )
        {
LABEL_75:
          v27 = v43;
          v29 = qword_4D03C50;
          goto LABEL_58;
        }
        v29 = qword_4D03C50;
        v27 = v43;
LABEL_51:
        v33 = *(unsigned __int16 *)(v29 + 18);
        v39 = *(_WORD *)(v29 + 18);
        *(_BYTE *)(v29 + 18) |= 0x80u;
        if ( !*(_QWORD *)(v27 + 144) || (a2[19].m128i_i8[10] & 0x10) != 0 )
        {
          v30 = 0;
          *(_QWORD *)(v27 + 144) = sub_6F6F40(a2, 0, v26, v33, v21, v22);
          v27 = v43;
        }
        sub_72A1A0(v27, v30, v26, v33, v21, v22);
        v34 = v43;
        LODWORD(v44) = 0;
        if ( (*(_BYTE *)(*(_QWORD *)(v43 + 144) - 8LL) & 1) == 0 && dword_4F07270[0] == unk_4F073B8 )
        {
          sub_7296F0((unsigned int)dword_4F04C64, &v44);
          v34 = v43;
        }
        sub_6E7AE0((__int64 *)(v34 + 144), v11, 0, 0, v41, a7, a8, v8);
        sub_729730((unsigned int)v44);
        v29 = qword_4D03C50;
        v27 = v43;
        *(_WORD *)(qword_4D03C50 + 18LL) = v39 & 0x180 | *(_WORD *)(qword_4D03C50 + 18LL) & 0xFE7F;
        goto LABEL_58;
      }
      if ( qword_4F077A8 )
        goto LABEL_41;
      v24 = qword_4D03C50;
      v25 = *(unsigned __int8 *)(qword_4D03C50 + 16LL);
    }
    else
    {
      v24 = qword_4D03C50;
      v25 = *(unsigned __int8 *)(qword_4D03C50 + 16LL);
      if ( (_DWORD)qword_4F077B4 )
        goto LABEL_33;
    }
LABEL_35:
    v42 = 1;
LABEL_36:
    if ( (unsigned __int8)v25 <= 3u && (*(_BYTE *)(v24 + 17) & 1) != 0 )
    {
      sub_6E68E0(0x1Cu, (__int64)a2);
    }
    else if ( !v41 || !(unsigned int)sub_8DAAE0(a2->m128i_i64[0], v11) )
    {
      v44 = (__int64 *)sub_6F6F40(a2, 0, v25, v23, v21, v22);
      sub_6E7AE0((__int64 *)&v44, v11, v9, v38, v41, a7, a8, v8);
      sub_6E70E0(v44, (__int64)a2);
    }
    goto LABEL_13;
  }
  if ( (unsigned __int8)v20 > 2u )
  {
    if ( v20 != 3 )
      sub_721090(v11);
    sub_6FC070(v11, a2, v41 == 0, 0, 0);
  }
  else
  {
    if ( !v20 )
      goto LABEL_15;
    v44 = (__int64 *)sub_6F6F40(a2, 0, v16, v17, v18, v19);
    sub_6E8160((__int64 *)&v44, v11, v9, v38, v41, a7, a8, 1, v8);
    sub_6E70E0(v44, (__int64)a2);
  }
LABEL_13:
  if ( dword_4D047EC && a2[1].m128i_i8[0] == 1 && (unsigned int)sub_8E3210(v11) )
    *(_BYTE *)(a2[9].m128i_i64[0] + 27) |= 1u;
LABEL_15:
  sub_6E4F10((__int64)a2, (__int64)v45, v41, 1);
  return sub_724E30(&v43);
}
