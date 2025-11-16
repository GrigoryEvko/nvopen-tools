// Function: sub_700E50
// Address: 0x700e50
//
__int64 __fastcall sub_700E50(
        __int64 a1,
        const __m128i *a2,
        const __m128i *a3,
        __int64 a4,
        __int64 a5,
        __m128i *a6,
        __int64 *a7,
        int a8,
        _QWORD *a9)
{
  unsigned __int8 v11; // r13
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r8
  bool v21; // zf
  __int64 v22; // rax
  char i; // dl
  __int64 v25; // rax
  char j; // dl
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rax
  char k; // dl
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r9
  char v38; // cl
  __int64 *v39; // rcx
  __int64 v40; // rcx
  char v41; // al
  __int64 v42; // rsi
  char v43; // al
  __int64 v44; // rsi
  unsigned int v45; // r13d
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rdi
  __m128i *v50; // rsi
  int v51; // edx
  char v52; // [rsp+8h] [rbp-1C8h]
  __int64 v53; // [rsp+8h] [rbp-1C8h]
  unsigned int v54; // [rsp+1Ch] [rbp-1B4h]
  int v55; // [rsp+24h] [rbp-1ACh] BYREF
  unsigned int v56; // [rsp+28h] [rbp-1A8h] BYREF
  int v57; // [rsp+2Ch] [rbp-1A4h] BYREF
  __int64 v58; // [rsp+30h] [rbp-1A0h] BYREF
  __int64 v59; // [rsp+38h] [rbp-198h] BYREF
  unsigned int v60[4]; // [rsp+40h] [rbp-190h] BYREF
  char v61; // [rsp+50h] [rbp-180h]
  __int64 v62; // [rsp+D0h] [rbp-100h]

  v11 = a1;
  v54 = a5;
  v58 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v18 = sub_724DC0(a1, a2, v14, v15, v16, v17);
  v21 = a2[1].m128i_i8[0] == 0;
  v59 = v18;
  if ( v21 )
    goto LABEL_5;
  v22 = a2->m128i_i64[0];
  for ( i = *(_BYTE *)(a2->m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v22 + 140) )
    v22 = *(_QWORD *)(v22 + 160);
  if ( !i || !a3[1].m128i_i8[0] )
    goto LABEL_5;
  v25 = a3->m128i_i64[0];
  for ( j = *(_BYTE *)(a3->m128i_i64[0] + 140); j == 12; j = *(_BYTE *)(v25 + 140) )
    v25 = *(_QWORD *)(v25 + 160);
  if ( !j )
    goto LABEL_5;
  if ( v54
    || (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x80200) == 0
    && (dword_4F077C4 == 2 || (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x10) == 0) )
  {
    v55 = 1;
    v56 = 0;
    goto LABEL_35;
  }
  v55 = 1;
  v56 = 0;
  if ( (unsigned __int8)(a1 - 50) <= 1u && (*(_BYTE *)(qword_4D03C50 + 18LL) & 8) == 0 )
  {
LABEL_35:
    if ( !word_4D04898 )
      goto LABEL_36;
    goto LABEL_41;
  }
  sub_6E6B60(a2, 0, (unsigned int)(a1 - 50), v19, v20, v54);
  sub_6E6B60(a3, 0, v27, v28, v29, v30);
  if ( a2[1].m128i_i8[0] == 2 )
  {
    if ( a3[1].m128i_i8[0] == 2 )
    {
      if ( (_BYTE)a1 == 119 )
      {
        sub_6E6260(a6);
        v55 = 0;
        goto LABEL_28;
      }
      sub_6E2E50(2, (__int64)a6);
      a6->m128i_i64[0] = a4;
      a6[1].m128i_i8[1] = 2;
      sub_6E5200(
        a1,
        (_DWORD)a2 + 144,
        (_DWORD)a3 + 144,
        a4,
        (_DWORD)a6 + 144,
        (__int64)&v55,
        (__int64)&v56,
        (__int64)a7);
      goto LABEL_27;
    }
    v42 = word_4D04898;
    if ( word_4D04898 )
    {
      v43 = ((*(_BYTE *)(qword_4D03C50 + 19LL) >> 1) ^ 1) & 1;
      if ( (_BYTE)a1 == 88 && v43 )
      {
        if ( (unsigned int)sub_70FCE0(&a2[9]) && !(unsigned int)sub_6E9820((__int64)a2, v42) )
        {
          sub_6E59E0((__int64)a3);
          v44 = 1;
          goto LABEL_71;
        }
        v52 = 30;
        goto LABEL_47;
      }
      if ( (_BYTE)a1 == 87 && v43 && (unsigned int)sub_70FCE0(&a2[9]) && (unsigned int)sub_6E9820((__int64)a2, v42) )
      {
        sub_6E59E0((__int64)a3);
        v44 = 0;
LABEL_71:
        sub_6E7080((__int64)a6, v44);
        sub_6FC3F0(a4, a6, 1u);
LABEL_72:
        v55 = 0;
        goto LABEL_28;
      }
    }
  }
  if ( !dword_4F077C0 )
  {
LABEL_48:
    if ( !dword_4F077BC || (_DWORD)qword_4F077B4 || qword_4F077A8 > 0xEA5Fu )
      goto LABEL_27;
    goto LABEL_24;
  }
  v52 = a1 - 58;
  if ( ((_BYTE)a1 == 52 || (unsigned __int8)(a1 - 58) <= 1u) && a2[1].m128i_i8[0] == 1 && a3[1].m128i_i8[0] == 1 )
  {
    if ( (unsigned int)sub_8D2E30(a2->m128i_i64[0])
      && (unsigned int)sub_8D2E30(a3->m128i_i64[0])
      && (unsigned int)sub_7164A0(a2[9].m128i_i64[0], v58, 0, 1, 0)
      && (unsigned int)sub_7164A0(a3[9].m128i_i64[0], v59, 0, 1, 0) )
    {
      sub_6E2E50(2, (__int64)a6);
      a6->m128i_i64[0] = a4;
      v51 = v59;
      a6[1].m128i_i8[1] = 2;
      sub_6E5200(a1, v58, v51, a4, (_DWORD)a6 + 144, (__int64)&v55, (__int64)&v56, (__int64)a7);
      goto LABEL_27;
    }
LABEL_47:
    if ( !dword_4F077C0 )
      goto LABEL_48;
  }
  if ( (_DWORD)qword_4F077B4 )
    goto LABEL_27;
  if ( qword_4F077A8 )
    goto LABEL_26;
  if ( !dword_4F077BC )
    goto LABEL_27;
LABEL_24:
  if ( !word_4D04898 )
  {
LABEL_43:
    if ( !v56 )
    {
LABEL_36:
      v33 = qword_4D03C50;
      if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u )
        goto LABEL_37;
      goto LABEL_57;
    }
LABEL_37:
    v34 = (__int64)a3;
    sub_6F7B90((__int64)a2, (__int64)a3, v11, a4, v54, a6);
    if ( a6[1].m128i_i8[0] == 1 )
    {
      v34 = (__int64)&v57;
      if ( !(unsigned int)sub_6DFA90(a6[9].m128i_i64[0], &v57, v60) )
      {
        v45 = v60[0];
        v34 = v60[0];
        if ( sub_6E53E0(5, v60[0], a7) )
        {
          v34 = (__int64)a7;
          sub_684B30(v45, a7);
        }
      }
    }
    if ( v56 )
      sub_6F4B70(a6, v34, v35, v36, v56, v37);
    goto LABEL_6;
  }
  v52 = a1 - 58;
LABEL_26:
  if ( (unsigned __int8)v52 <= 1u && a2[1].m128i_i8[0] == 1 )
  {
    v46 = a2[9].m128i_i64[0];
    if ( *(_BYTE *)(v46 + 24) == 3 && (*(_BYTE *)(v46 + 25) & 3) == 0 && a3[1].m128i_i8[0] == 1 )
    {
      v47 = a3[9].m128i_i64[0];
      if ( *(_BYTE *)(v47 + 24) == 3 && (*(_BYTE *)(v47 + 25) & 3) == 0 )
      {
        v48 = *(_QWORD *)(v46 + 56);
        if ( v48 == *(_QWORD *)(v47 + 56) )
        {
          v53 = v48;
          if ( !(unsigned int)sub_8D2A90(*(_QWORD *)(v48 + 120)) )
          {
            v49 = *(_QWORD *)(v53 + 120);
            if ( (*(_BYTE *)(v49 + 140) & 0xFB) != 8 || (sub_8D4C10(v49, dword_4F077C4 != 2) & 2) == 0 )
            {
              sub_6E2E50(2, (__int64)a6);
              a6->m128i_i64[0] = a4;
              v50 = a6 + 9;
              a6[1].m128i_i8[1] = 2;
              if ( v11 == 58 )
                sub_72BB90(a4, v50);
              else
                sub_72BB40(a4, v50);
              goto LABEL_72;
            }
          }
        }
      }
    }
  }
LABEL_27:
  if ( v55 )
  {
    if ( !word_4D04898 )
      goto LABEL_43;
LABEL_41:
    v33 = qword_4D03C50;
    v38 = *(_BYTE *)(qword_4D03C50 + 16LL);
    if ( (unsigned __int8)(v38 - 1) > 1u )
    {
      if ( v38 != 3 )
        goto LABEL_43;
      v39 = *(__int64 **)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 624);
      if ( !v39
        || (v40 = *v39) == 0
        || ((*(_BYTE *)(v40 + 80) - 7) & 0xFD) != 0
        || (*(_BYTE *)(*(_QWORD *)(v40 + 88) + 170LL) & 0x20) == 0 )
      {
        if ( !v56 )
        {
LABEL_57:
          v41 = *(_BYTE *)(v33 + 17);
          if ( (v41 & 1) == 0 || word_4D04898 | v54 || (v41 & 4) != 0 )
            goto LABEL_37;
          if ( (unsigned int)sub_6E5430() )
            sub_6851C0(0x1Cu, a7);
LABEL_5:
          sub_6E6260(a6);
          goto LABEL_6;
        }
        goto LABEL_37;
      }
    }
    if ( a2[1].m128i_i8[0] == 2 && a2[19].m128i_i8[13] == 12
      || a3[1].m128i_i8[0] == 2 && a3[19].m128i_i8[13] == 12
      || (unsigned int)sub_8DBE70(a2->m128i_i64[0])
      || (unsigned int)sub_8DBE70(a3->m128i_i64[0]) )
    {
      v56 = 1;
      goto LABEL_37;
    }
    goto LABEL_43;
  }
LABEL_28:
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) )
  {
    sub_6F7CB0((__int64)a2, (__int64)a3, v11, a4, v60);
    if ( v61 )
    {
      v31 = *(_QWORD *)v60;
      for ( k = *(_BYTE *)(*(_QWORD *)v60 + 140LL); k == 12; k = *(_BYTE *)(v31 + 140) )
        v31 = *(_QWORD *)(v31 + 160);
      if ( k )
        a6[18].m128i_i64[0] = v62;
    }
  }
LABEL_6:
  a6[4].m128i_i8[0] = a3[4].m128i_i8[0] | a2[4].m128i_i8[0];
  sub_6E3BA0((__int64)a6, a7, a8, a9);
  sub_724E30(&v58);
  return sub_724E30(&v59);
}
