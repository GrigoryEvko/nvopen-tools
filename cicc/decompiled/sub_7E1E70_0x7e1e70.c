// Function: sub_7E1E70
// Address: 0x7e1e70
//
__int64 *__fastcall sub_7E1E70(__int64 *a1)
{
  __int64 *v1; // r12
  const __m128i *v2; // rax
  __int64 v3; // rdi
  __int64 v4; // rax
  _QWORD *v5; // rax
  __int64 v7; // rax
  const __m128i *v8; // [rsp+8h] [rbp-18h] BYREF

  v1 = a1;
  v2 = (const __m128i *)sub_724DC0();
  v3 = *a1;
  v8 = v2;
  if ( (unsigned int)sub_8D2780(v3) )
  {
    v7 = sub_6E8500(v1);
    v1 = (__int64 *)sub_73E130(v1, v7);
  }
  v4 = sub_7E1E20(*v1);
  sub_72BB40(v4, v8);
  v5 = sub_73A720(v8, (__int64)v8);
  v1[2] = (__int64)v5;
  *(_BYTE *)(v5[7] - 8LL) &= ~8u;
  sub_724E30((__int64)&v8);
  return v1;
}
