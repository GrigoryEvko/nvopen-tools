// Function: sub_22074A0
// Address: 0x22074a0
//
__int64 __fastcall sub_22074A0(__int64 *a1)
{
  __int64 result; // rax
  void (__fastcall *v2)(__int64); // rax

  result = *a1;
  if ( *a1 && !_InterlockedSub((volatile signed __int32 *)(result - 128), 1u) )
  {
    v2 = *(void (__fastcall **)(__int64))(result - 104);
    if ( v2 )
      v2(*a1);
    result = sub_22527D0(*a1);
    *a1 = 0;
  }
  return result;
}
