// Function: sub_7E0CD0
// Address: 0x7e0cd0
//
_QWORD *__fastcall sub_7E0CD0(const __m128i *a1, __int64 a2)
{
  _QWORD *v3; // r14
  int v4; // eax
  bool v5; // bl
  const __m128i *v6; // rsi
  _QWORD *v7; // r15
  _BYTE *v8; // rbx
  char v9; // r15
  __int8 v10; // r13
  __int64 v12; // rax
  _BYTE *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // [rsp+8h] [rbp-48h]
  const __m128i *v19; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(_QWORD **)a2;
  v19 = (const __m128i *)sub_724DC0();
  v4 = sub_8D33B0(v3);
  v5 = v4 == 0;
  if ( (*(_BYTE *)(a2 + 25) & 1) != 0 || !v4 )
  {
    v12 = sub_72D2E0(v3);
    v6 = v19;
    sub_72BB40(v12, v19);
    v7 = sub_73A720(v19, (__int64)v19);
    if ( (*(_BYTE *)(a2 + 25) & 1) == 0 && !v5 )
      goto LABEL_4;
  }
  else
  {
    v6 = v19;
    sub_72BB40((__int64)v3, v19);
    v7 = sub_73A720(v19, (__int64)v19);
    if ( (*(_BYTE *)(a2 + 25) & 1) == 0 )
      goto LABEL_4;
  }
  v13 = sub_73DCD0(v7);
  v7 = v13;
  if ( (*(_BYTE *)(a2 + 25) & 1) == 0 )
    v7 = sub_731370((__int64)v13, (__int64)v6, v14, v15, v16, v17);
LABEL_4:
  v8 = sub_730FF0(a1);
  sub_7304E0((__int64)v8);
  v8[27] |= 2u;
  *((_QWORD *)v8 + 2) = v7;
  v9 = *(_BYTE *)(a2 + 25);
  v10 = a1[1].m128i_i8[9];
  v18 = a1[1].m128i_i64[0];
  sub_7266C0((__int64)a1, 1);
  a1[1].m128i_i64[0] = v18;
  a1[1].m128i_i8[9] = a1[1].m128i_i8[9] & 0xFB | (4 * ((v10 & 4) != 0));
  sub_73D8E0((__int64)a1, 0x5Bu, (__int64)v3, v9 & 1, (__int64)v8);
  return sub_724E30((__int64)&v19);
}
