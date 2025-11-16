// Function: sub_23569D0
// Address: 0x23569d0
//
__int64 __fastcall sub_23569D0(volatile signed __int32 *a1)
{
  __int64 result; // rax

  if ( !_InterlockedSub(a1, 1u) )
    return (*(__int64 (__fastcall **)(volatile signed __int32 *))(*((_QWORD *)a1 - 1) + 8LL))(a1 - 2);
  return result;
}
