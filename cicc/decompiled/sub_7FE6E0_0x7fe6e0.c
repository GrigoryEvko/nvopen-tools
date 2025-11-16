// Function: sub_7FE6E0
// Address: 0x7fe6e0
//
_QWORD *__fastcall sub_7FE6E0(__int64 a1, __int64 a2, char a3, const __m128i *a4, __m128i *a5)
{
  const __m128i *v7; // r12
  __m128i *v8; // r15
  __int64 v9; // rbx
  const __m128i *v10; // rax
  _QWORD *v11; // r15
  _QWORD *v12; // rax
  __int64 v14; // rdi
  __int64 i; // rax
  __int64 j; // rsi
  __m128i *v17; // rsi
  __int64 v18; // rsi
  __m128i *v20; // [rsp+8h] [rbp-48h]
  const __m128i *v21; // [rsp+18h] [rbp-38h] BYREF

  v20 = (__m128i *)sub_7F98A0(a2, 0);
  v7 = (const __m128i *)sub_7F9160(a2);
  if ( !a1 )
  {
    sub_7F6D70(0, *(_BYTE **)(a2 + 8));
    if ( !v7 )
      BUG();
    v9 = sub_7F9D60();
    v10 = (const __m128i *)sub_724DC0();
    v21 = v10;
    goto LABEL_17;
  }
  v8 = sub_7FDF40(a1, 2 - a3, 0);
  sub_7F6D70(a1, *(_BYTE **)(a2 + 8));
  if ( v7 )
  {
    v9 = sub_7F9D60();
    v10 = (const __m128i *)sub_724DC0();
    v21 = v10;
    if ( v8 )
    {
      v11 = sub_731330((__int64)v8);
LABEL_5:
      sub_724E30((__int64)&v21);
      v12 = sub_7F9E20(v20, v7, (__int64)v11, 0);
      return sub_7E6A50(v12, a5->m128i_i32);
    }
LABEL_17:
    v18 = (__int64)v10;
    sub_72BB40(v9, v10);
    v11 = sub_73A720(v21, v18);
    goto LABEL_5;
  }
  v14 = v8[9].m128i_i64[1];
  for ( i = v14; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( !*(_QWORD *)(*(_QWORD *)(i + 168) + 40LL) )
    BUG();
  for ( j = sub_8D71D0(v14); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v17 = (__m128i *)sub_73E130(v20, j);
  if ( (((v8[12].m128i_i8[13] & 0x1C) - 8) & 0xF4) == 0
    && (*(_BYTE *)(*(_QWORD *)(v8[2].m128i_i64[1] + 32) + 176LL) & 0x10) != 0 )
  {
    v7 = a4;
  }
  v17[1].m128i_i64[0] = (__int64)v7;
  return sub_7F88F0((__int64)v8, v17, 0, a5);
}
