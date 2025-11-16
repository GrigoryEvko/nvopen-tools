// Function: sub_740980
// Address: 0x740980
//
_DWORD *__fastcall sub_740980(const __m128i *a1)
{
  __m128i *v2; // r12
  __int64 v3; // rdi
  _QWORD *v4; // r13
  int v5; // eax
  unsigned __int8 v6; // dl
  __int64 i; // rax
  _BYTE *v8; // rax
  int v10; // [rsp+4h] [rbp-2Ch] BYREF
  const __m128i *v11; // [rsp+8h] [rbp-28h] BYREF

  v2 = sub_740630(a1);
  v11 = (const __m128i *)sub_724DC0();
  sub_7296C0(&v10);
  sub_724C70((__int64)a1, 12);
  v3 = v2[8].m128i_i64[0];
  a1[8].m128i_i64[0] = v3;
  if ( !(unsigned int)sub_8DBE70(v3) )
    v2[8].m128i_i64[0] = dword_4D03B80;
  a1[11].m128i_i8[0] = 1;
  v4 = sub_73A720(v2, 12);
  v5 = sub_8D2930(v2[8].m128i_i64[0]);
  v6 = 5;
  if ( v5 )
  {
    for ( i = v2[8].m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v6 = *(_BYTE *)(i + 160);
  }
  sub_72BAF0((__int64)v11, 1, v6);
  v4[2] = sub_73A720(v11, 1);
  v8 = sub_73DBF0(0x27u, a1[8].m128i_i64[0], (__int64)v4);
  a1[11].m128i_i64[1] = (__int64)v8;
  v8[27] |= 2u;
  sub_724E30((__int64)&v11);
  return sub_729730(v10);
}
