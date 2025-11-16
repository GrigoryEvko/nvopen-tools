// Function: sub_68B170
// Address: 0x68b170
//
__int64 __fastcall sub_68B170(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a1 + 16);
  if ( (_BYTE)result == 1 )
  {
    result = *(_QWORD *)(a1 + 144);
    *(_BYTE *)(result + 25) |= 0x20u;
  }
  else if ( (_BYTE)result == 2 )
  {
    result = *(_QWORD *)(a1 + 288);
    if ( result )
    {
      *(_BYTE *)(result + 25) |= 0x20u;
    }
    else
    {
      result = unk_4D03C50;
      if ( *(_BYTE *)(unk_4D03C50 + 16LL) )
      {
        result = sub_6ED0D0();
        *(_QWORD *)(a1 + 288) = result;
        if ( result )
          *(_BYTE *)(result + 25) |= 0x20u;
      }
    }
  }
  return result;
}
