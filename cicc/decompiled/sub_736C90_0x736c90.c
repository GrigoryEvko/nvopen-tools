// Function: sub_736C90
// Address: 0x736c90
//
__int64 __fastcall sub_736C90(__int64 a1, int a2)
{
  char v2; // bl
  __int64 result; // rax

  if ( a2 )
  {
    if ( dword_4F077C4 == 2
      && (unk_4F07778 > 201102 || dword_4F07774)
      && dword_4D04964
      && (*(_WORD *)(a1 + 192) & 0x3080) == 0x2000 )
    {
      v2 = 0;
      sub_686A30(8u, 0x677u, dword_4F07508, (_QWORD *)(*(_QWORD *)a1 + 48LL), *(_QWORD *)a1);
    }
    else
    {
      v2 = a2 & 1;
      if ( (*(_BYTE *)(a1 + 196) & 0x40) != 0
        && HIDWORD(qword_4F077B4)
        && (*(_BYTE *)(a1 + 89) & 4) != 0
        && (*(_BYTE *)(a1 + 203) & 1) == 0 )
      {
        sub_736C60(28, *(__int64 **)(a1 + 104));
      }
    }
  }
  else
  {
    *(_BYTE *)(a1 + 203) &= 0x9Fu;
    v2 = 0;
  }
  result = *(_BYTE *)(a1 + 192) & 0x7F;
  *(_BYTE *)(a1 + 192) = result | (v2 << 7);
  return result;
}
