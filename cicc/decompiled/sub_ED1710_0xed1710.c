// Function: sub_ED1710
// Address: 0xed1710
//
__int64 __fastcall sub_ED1710(__int64 a1, __int64 a2)
{
  bool v2; // zf
  __int64 result; // rax
  char v4; // al
  char v5; // cl

  v2 = !sub_ED1700(a1);
  result = *(unsigned __int8 *)(a2 + 32);
  if ( v2 )
  {
    v5 = result & 0xF;
    if ( (unsigned int)(result & 0xF) - 7 > 1 )
    {
      result = result & 0xFFFFFFCF | 0x10;
      *(_BYTE *)(a2 + 32) = result;
      if ( v5 != 9 )
        *(_BYTE *)(a2 + 33) |= 0x40u;
    }
  }
  else
  {
    v4 = result & 0xCF | 0x20;
    *(_BYTE *)(a2 + 32) = v4;
    result = v4 & 0xF;
    if ( (_BYTE)result != 9 )
      *(_BYTE *)(a2 + 33) |= 0x40u;
  }
  return result;
}
