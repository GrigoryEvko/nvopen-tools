// Function: sub_7F7FC0
// Address: 0x7f7fc0
//
_QWORD *__fastcall sub_7F7FC0(__int64 a1)
{
  __int64 v1; // r13
  const __m128i *v2; // rax
  _QWORD *v3; // r12
  __int64 v5; // rsi
  __int64 v6; // r14
  __int64 v7; // rbx
  _QWORD *v8; // rax
  _QWORD *v9; // r13
  const __m128i *v10; // [rsp+8h] [rbp-28h] BYREF

  v1 = qword_4F18B08;
  if ( !qword_4F18B08 )
  {
    v6 = sub_7E1C10();
    v7 = sub_72CBE0();
    v8 = sub_7259C0(7);
    v8[20] = v7;
    v9 = v8;
    *(_BYTE *)(v8[21] + 16LL) = (2 * (dword_4F06968 == 0)) | *(_BYTE *)(v8[21] + 16LL) & 0xFD;
    if ( v6 )
      *(_QWORD *)v8[21] = sub_724EF0(v6);
    qword_4F18B08 = sub_72D2E0(v9);
    v1 = qword_4F18B08;
  }
  v2 = (const __m128i *)sub_724DC0();
  v10 = v2;
  if ( a1 )
  {
    v3 = sub_731330(a1);
  }
  else
  {
    v5 = (__int64)v2;
    sub_72BB40(v1, v2);
    v3 = sub_73A720(v10, v5);
  }
  sub_724E30((__int64)&v10);
  return v3;
}
