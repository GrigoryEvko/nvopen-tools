// Function: sub_6F50A0
// Address: 0x6f50a0
//
__int64 __fastcall sub_6F50A0(__int64 a1, __m128i *a2, int a3, const __m128i *a4, __int64 a5, __int64 a6)
{
  FILE *v6; // r15
  unsigned int v9; // r13d
  __int64 v10; // rax
  char i; // dl
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 result; // rax
  int v17; // r9d
  FILE *v18; // rsi
  _DWORD *v19; // rax
  const __m128i *v20; // rcx
  __int64 v21; // r12
  int v22; // eax
  _DWORD *v23; // rax
  const __m128i *v24; // [rsp+8h] [rbp-38h]
  const __m128i *v25; // [rsp+8h] [rbp-38h]
  const __m128i *v26; // [rsp+8h] [rbp-38h]

  v6 = (FILE *)a5;
  if ( !a1 || (v9 = 28, (*(_BYTE *)(a1 + 174) & 0xFB) == 0) )
    v9 = word_4D04898 == 0 ? 59 : 2404;
  if ( a2 )
  {
    if ( !a2[1].m128i_i8[0] )
      return 0;
    v10 = a2->m128i_i64[0];
    for ( i = *(_BYTE *)(a2->m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v10 + 140) )
      v10 = *(_QWORD *)(v10 + 160);
    if ( !i )
      return 0;
  }
  if ( !qword_4D03C50 )
  {
    v13 = word_4D04898;
    if ( !word_4D04898 )
      return 0;
    goto LABEL_23;
  }
  v24 = a4;
  v12 = sub_6E4B50();
  a4 = v24;
  if ( v12 )
  {
    if ( !a1 || (*(_BYTE *)(a1 + 193) & 2) != 0 )
      return 0;
    if ( (*(_BYTE *)(a1 + 202) & 4) != 0 )
    {
      if ( dword_4F077BC )
      {
        if ( !(_DWORD)qword_4F077B4 )
          return 0;
        goto LABEL_33;
      }
      if ( (_DWORD)qword_4F077B4 )
      {
LABEL_33:
        if ( (*(_BYTE *)(a1 + 195) & 3) == 1 )
          return 0;
      }
    }
    else if ( dword_4F077BC | (unsigned int)qword_4F077B4 )
    {
      goto LABEL_33;
    }
    if ( unk_4D041F8 )
    {
      if ( (unsigned int)sub_7176C0(a1, 0) )
        return 0;
      a4 = v24;
    }
  }
  v13 = word_4D04898;
  v14 = qword_4D03C50;
  if ( !word_4D04898 )
  {
    if ( qword_4D03C50 && (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0 )
    {
LABEL_26:
      if ( a2 )
      {
        sub_6E68E0(v9, (__int64)a2);
        return 1;
      }
      else
      {
        v17 = sub_6E5430();
        result = 1;
        if ( v17 )
        {
          sub_6851C0(v9, v6);
          return 1;
        }
      }
      return result;
    }
    return 0;
  }
  if ( qword_4D03C50 )
  {
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u && (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) != 0 )
    {
      a4 = (const __m128i *)qword_4F04C68;
      v15 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( *(_BYTE *)(v15 + 4) == 8 )
      {
LABEL_16:
        if ( a2 && ((*(_BYTE *)(v15 + 12) & 0x10) == 0 || (unsigned int)sub_696840((__int64)a2)) )
          sub_6F4B70(a2, v15, v13, (__int64)a4, a5, a6);
        return 0;
      }
    }
LABEL_25:
    if ( (*(_BYTE *)(v14 + 19) & 0x40) == 0 )
    {
      if ( !a3 && (_DWORD)v13 && (*(_BYTE *)(v14 + 17) & 5) == 1 && (!a1 || (*(_BYTE *)(a1 + 193) & 2) == 0) )
      {
        *(_BYTE *)(v14 + 19) |= 0x20u;
        return 0;
      }
      return 0;
    }
    goto LABEL_26;
  }
LABEL_23:
  v15 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( *(_BYTE *)(v15 + 4) == 8 )
    goto LABEL_16;
  v14 = qword_4D03C50;
  if ( qword_4D03C50 )
    goto LABEL_25;
  if ( !(_DWORD)v13 || a3 )
    return 0;
  if ( a1 && *(_BYTE *)(a1 + 174) == 1 && (v26 = a4, v22 = sub_72F310(a1, 1, v13, a4, a5, a6), a4 = v26, v22) )
  {
    if ( a2 )
      v6 = (FILE *)((char *)a2[4].m128i_i64 + 4);
    v18 = v6;
    v23 = sub_67DA80(0x960u, v6, *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL));
    v20 = v26;
    v21 = (__int64)v23;
  }
  else
  {
    v25 = a4;
    if ( a2 )
      v6 = (FILE *)((char *)a2[4].m128i_i64 + 4);
    v18 = v6;
    v19 = sub_67D9D0(v9, v6);
    v20 = v25;
    v21 = (__int64)v19;
  }
  if ( v20 )
  {
    v18 = (FILE *)v20;
    sub_67E370(v21, v20);
  }
  sub_685910(v21, v18);
  return 1;
}
