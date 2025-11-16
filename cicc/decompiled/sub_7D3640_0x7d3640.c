// Function: sub_7D3640
// Address: 0x7d3640
//
const __m128i *__fastcall sub_7D3640(const char *a1, __int64 a2, __int64 a3)
{
  const __m128i *result; // rax
  _QWORD v4[10]; // [rsp+0h] [rbp-50h] BYREF

  sub_87AB50(a2, v4, a3);
  result = (const __m128i *)sub_7D2AC0(v4, a1, 0x10u);
  if ( !result )
    return sub_7D2920((__int64)v4, (__int64)a1);
  return result;
}
