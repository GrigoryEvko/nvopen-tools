// Function: sub_303E610
// Address: 0x303e610
//
__int64 __fastcall sub_303E610(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // ebx
  __int64 result; // rax

  v4 = sub_AE5020(a4, a3);
  if ( (unsigned __int8)v4 >= 7u )
    v4 = 7;
  if ( !a2
    || (*(_BYTE *)(a2 + 32) & 0xFu) - 7 > 1
    || (unsigned __int8)sub_CE9220(a2)
    || (unsigned __int8)sub_B2DDD0(a2, 0, 0, 1, 1, 0, 0) )
  {
    return v4;
  }
  result = 4;
  if ( (unsigned __int8)v4 > 4u )
    return v4;
  return result;
}
