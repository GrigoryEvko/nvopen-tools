// Function: sub_7E32B0
// Address: 0x7e32b0
//
__m128i *__fastcall sub_7E32B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  int v4; // eax
  __int64 v5; // r15
  int v6; // r9d
  __m128i *v7; // r12
  _QWORD *v8; // rax
  __int64 v9; // r12
  const __m128i *v10; // rax
  __int64 v12; // rax
  int v13; // [rsp+4h] [rbp-3Ch]

  v3 = *(_QWORD *)(a1 + 168);
  v4 = sub_7DB880(a1);
  v5 = v4;
  if ( v4 == 11 || a2 )
  {
    v6 = 0;
  }
  else
  {
    v6 = 1;
    v7 = (__m128i *)qword_4D03EE0[v4];
    if ( v7 )
    {
      if ( !a3 )
      {
LABEL_5:
        *(_QWORD *)(v3 + 192) = v7;
        goto LABEL_12;
      }
      goto LABEL_12;
    }
  }
  v13 = v6;
  v8 = sub_7259C0(8);
  v8[22] = 0;
  v9 = (__int64)v8;
  v10 = (const __m128i *)sub_7E1330();
  *(_QWORD *)(v9 + 160) = sub_73C570(v10, 1);
  sub_8D6090(v9);
  v7 = sub_7E2190(0, 1, v9, 1);
  if ( unk_4D04894 && (v12 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 168) + 152LL) + 144LL)) != 0 )
  {
    while ( (*(_BYTE *)(v12 + 193) & 4) == 0 || *(_BYTE *)(v12 + 174) != 1 || (*(_BYTE *)(v12 + 88) & 4) == 0 )
    {
      v12 = *(_QWORD *)(v12 + 112);
      if ( !v12 )
        goto LABEL_8;
    }
  }
  else
  {
LABEL_8:
    v7[5].m128i_i8[8] &= ~4u;
  }
  v7[10].m128i_i8[8] = *(_BYTE *)(v3 + 109) & 7 | v7[10].m128i_i8[8] & 0xF8;
  if ( v13 )
    qword_4D03EE0[v5] = v7;
  if ( !a3 )
  {
    if ( a2 )
      sub_721090();
    goto LABEL_5;
  }
LABEL_12:
  sub_7E1230(v7, 0, HIDWORD(qword_4D045BC) == 0, 1);
  v7[9].m128i_i8[12] |= 0x40u;
  return v7;
}
