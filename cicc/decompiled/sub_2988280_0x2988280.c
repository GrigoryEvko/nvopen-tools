// Function: sub_2988280
// Address: 0x2988280
//
__int64 __fastcall sub_2988280(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 48);
  if ( *(_QWORD *)(a1 + 112) == result )
    return 0;
  return result;
}
