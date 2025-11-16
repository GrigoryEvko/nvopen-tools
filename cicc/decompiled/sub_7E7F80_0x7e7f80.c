// Function: sub_7E7F80
// Address: 0x7e7f80
//
_QWORD *__fastcall sub_7E7F80(const __m128i *a1, unsigned int a2, _DWORD *a3, unsigned int a4)
{
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD *result; // rax
  const __m128i *v10; // r13
  __m128i *v11; // r13
  const __m128i *v12; // rax
  _QWORD *v13; // rax
  __m128i *v14; // [rsp+8h] [rbp-28h] BYREF

  *a3 = 0;
  if ( (unsigned int)sub_7E2550((__int64)a1, &v14) )
  {
    result = sub_73E830((__int64)v14);
    *a3 = 1;
  }
  else if ( (unsigned int)sub_731920((__int64)a1, a2, a4, v6, v7, v8) )
  {
    return sub_73B8B0(a1, 0);
  }
  else
  {
    v10 = (const __m128i *)sub_730FF0(a1);
    if ( (a1[1].m128i_i8[9] & 1) != 0 )
    {
      v11 = (__m128i *)sub_73E1B0((__int64)v10, a2);
      v14 = sub_7E7ED0(v11);
      v12 = (const __m128i *)sub_73DCD0(v11);
      sub_730620((__int64)a1, v12);
      v13 = sub_73E830((__int64)v14);
      result = sub_73DCD0(v13);
    }
    else
    {
      v14 = sub_7E7ED0(v10);
      sub_730620((__int64)a1, v10);
      result = sub_73E830((__int64)v14);
    }
    *a3 = 1;
  }
  return result;
}
