// Function: sub_DD8750
// Address: 0xdd8750
//
__int64 __fastcall sub_DD8750(__int64 *a1, __int64 a2, unsigned __int8 *a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax

  result = sub_DD8590(a1, a2, a3, a4, a5);
  if ( !(_BYTE)result )
    return sub_DB6680((__int64)a1, a2, (__int64)a3);
  return result;
}
