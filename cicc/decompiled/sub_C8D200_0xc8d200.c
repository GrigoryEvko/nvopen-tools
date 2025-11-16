// Function: sub_C8D200
// Address: 0xc8d200
//
__int64 __fastcall sub_C8D200(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax

  if ( a2 || (result = malloc(1, 0, a3, a4, a5, a6)) == 0 )
    sub_C64F00("Allocation failed", 1u);
  return result;
}
