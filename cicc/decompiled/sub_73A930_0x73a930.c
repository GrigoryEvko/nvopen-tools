// Function: sub_73A930
// Address: 0x73a930
//
__int64 __fastcall sub_73A930(_QWORD *a1)
{
  __int64 v1; // r13
  _QWORD *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  _UNKNOWN *__ptr32 *v5; // r8
  __int64 v6; // rax
  const __m128i *v7; // rsi
  _QWORD *v8; // rax
  __int64 v9; // r12
  const __m128i *v11; // [rsp+8h] [rbp-18h] BYREF

  v11 = (const __m128i *)sub_724DC0();
  v1 = sub_72D2E0(a1);
  if ( (unsigned int)sub_8DBE70(a1) )
  {
    v2 = sub_72BA30(5u);
    sub_72BB40((__int64)v2, v11);
    v6 = sub_73A460(v11, (__int64)v11, v3, v4, v5);
    v7 = v11;
    sub_70FDD0(v6, (__int64)v11, v1, 0);
  }
  else
  {
    v7 = v11;
    sub_72BB40(v1, v11);
  }
  v8 = sub_73A720(v11, (__int64)v7);
  v9 = sub_73DCD0(v8);
  sub_724E30((__int64)&v11);
  return v9;
}
