// Function: sub_38AF780
// Address: 0x38af780
//
__int64 __fastcall sub_38AF780(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        unsigned __int8 a4,
        unsigned __int8 a5,
        __m128 a6,
        double a7,
        double a8)
{
  __int64 result; // rax

  result = sub_388AF10(a1, 12, "expected '(' in call");
  if ( !(_BYTE)result )
    return sub_38AF420(a1, a2, a3, a4, a5, a6, a7, a8);
  return result;
}
