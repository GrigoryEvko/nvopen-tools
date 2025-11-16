// Function: sub_8D1280
// Address: 0x8d1280
//
__int64 __fastcall sub_8D1280(__int64 a1, int a2)
{
  __int64 result; // rax

  result = (unsigned int)*(unsigned __int8 *)(a1 + 140) - 9;
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u && !a2 )
  {
    result = *(_QWORD *)(a1 + 168);
    *(_BYTE *)(result + 112) |= 4u;
  }
  return result;
}
