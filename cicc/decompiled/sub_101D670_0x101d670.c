// Function: sub_101D670
// Address: 0x101d670
//
__int64 __fastcall sub_101D670(unsigned __int8 *a1, _BYTE *a2, char a3, __m128i *a4)
{
  __int64 result; // rax

  result = sub_101D570(0x1Bu, (__int64)a1, a2, a3, a4, 3);
  if ( !result )
    return sub_1004A20(a1, (__int64)a2, (__int64)a4);
  return result;
}
