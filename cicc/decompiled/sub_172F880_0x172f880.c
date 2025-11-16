// Function: sub_172F880
// Address: 0x172f880
//
unsigned __int8 *__fastcall sub_172F880(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        double a5,
        double a6,
        double a7)
{
  int v8; // edx
  int v9; // eax
  unsigned __int8 *result; // rax

  v9 = *(unsigned __int16 *)(a2 + 18);
  v8 = *(unsigned __int16 *)(a3 + 18);
  BYTE1(v9) &= ~0x80u;
  BYTE1(v8) &= ~0x80u;
  if ( v8 != v9 )
    return sub_172E860(a1, a2, a3, a5, a6, a7);
  if ( v9 != 32 )
    return sub_172E860(a1, a2, a3, a5, a6, a7);
  result = sub_172AD80(a1, a2, a3, 0, a4, a5, a6, a7);
  if ( !result )
    return sub_172E860(a1, a2, a3, a5, a6, a7);
  return result;
}
