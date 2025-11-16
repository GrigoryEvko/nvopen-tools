// Function: sub_B8B500
// Address: 0xb8b500
//
__int64 __fastcall sub_B8B500(__int64 a1, __int64 *a2, unsigned __int8 a3)
{
  __int64 result; // rax

  result = sub_B7F780((__int64)a2, a3);
  if ( (_BYTE)result )
    return sub_B8B080(*(_QWORD *)(a1 + 8) + 568LL, a2);
  return result;
}
