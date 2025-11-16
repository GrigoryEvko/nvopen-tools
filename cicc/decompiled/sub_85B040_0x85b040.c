// Function: sub_85B040
// Address: 0x85b040
//
__int64 __fastcall sub_85B040(__int64 a1, int a2)
{
  __int64 result; // rax

  result = (unsigned int)*(unsigned __int8 *)(a1 + 140) - 9;
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u && !a2 )
  {
    result = *(_QWORD *)(a1 + 168);
    *(_BYTE *)(result + 112) |= 2u;
  }
  return result;
}
