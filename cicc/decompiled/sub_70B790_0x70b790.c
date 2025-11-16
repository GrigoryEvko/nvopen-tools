// Function: sub_70B790
// Address: 0x70b790
//
const __m128i *__fastcall sub_70B790(unsigned __int8 a1, const __m128i *a2, __m128i *a3, __int64 a4)
{
  __int64 v5; // rdx
  const __m128i *result; // rax
  unsigned __int8 v7; // [rsp+7h] [rbp-39h] BYREF
  _BYTE v8[4]; // [rsp+8h] [rbp-38h] BYREF
  _BYTE v9[4]; // [rsp+Ch] [rbp-34h] BYREF
  _QWORD v10[6]; // [rsp+10h] [rbp-30h] BYREF

  sub_70FEF0(a4, &v7, v8, v9);
  v10[0] = sub_709B30(a1, a2);
  v10[1] = v5;
  if ( (unsigned __int8)sub_12F9B50(v10, &unk_4F07870) )
    result = (const __m128i *)(16LL * v7 + 82864032);
  else
    result = (const __m128i *)(16LL * v7 + 82863808);
  *a3 = _mm_loadu_si128(result);
  return result;
}
