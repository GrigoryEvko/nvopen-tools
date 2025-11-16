// Function: sub_16AF4C0
// Address: 0x16af4c0
//
__int64 __fastcall sub_16AF4C0(volatile signed __int32 *a1, signed __int32 a2, signed __int32 a3)
{
  return (unsigned int)_InterlockedCompareExchange(a1, a2, a3);
}
