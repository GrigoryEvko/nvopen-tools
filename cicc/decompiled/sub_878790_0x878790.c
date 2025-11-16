// Function: sub_878790
// Address: 0x878790
//
__int64 __fastcall sub_878790(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a1 + 18);
  *(_BYTE *)(a1 + 16) &= 0xF8u;
  if ( (result & 2) != 0 )
  {
    result = (unsigned int)result & 0xFFFFFFFD;
    *(_QWORD *)(a1 + 32) = 0;
    *(_BYTE *)(a1 + 18) = result;
  }
  else
  {
    *(_QWORD *)(a1 + 32) = 0;
  }
  return result;
}
