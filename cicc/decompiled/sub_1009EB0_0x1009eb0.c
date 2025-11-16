// Function: sub_1009EB0
// Address: 0x1009eb0
//
unsigned __int8 *__fastcall sub_1009EB0(_BYTE *a1, _BYTE *a2, char a3, __m128i *a4, char a5, char a6)
{
  unsigned __int8 *result; // rax
  _BYTE *v11; // [rsp+0h] [rbp-30h] BYREF
  _BYTE *v12; // [rsp+8h] [rbp-28h] BYREF

  v11 = a1;
  v12 = a2;
  if ( !a5 && a6 == 1 )
  {
    result = (unsigned __int8 *)sub_FFE3E0(0x12u, &v11, &v12, a4->m128i_i64);
    if ( result )
      return result;
    a2 = v12;
    a1 = v11;
  }
  return sub_1009850((__int64)a1, (__int64)a2, a3, a4, a5, a6);
}
