// Function: sub_6440B0
// Address: 0x6440b0
//
__int64 __fastcall sub_6440B0(unsigned __int8 a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_736C60(a1, *(_QWORD *)(a2 + 184));
  if ( !result )
    return sub_736C60(a1, *(_QWORD *)(a2 + 200));
  return result;
}
