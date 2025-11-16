// Function: sub_1DDC510
// Address: 0x1ddc510
//
__int64 __fastcall sub_1DDC510(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 232);
  if ( result )
    return *(_QWORD *)(result + 128);
  return result;
}
