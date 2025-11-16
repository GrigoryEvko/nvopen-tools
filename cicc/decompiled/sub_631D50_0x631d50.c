// Function: sub_631D50
// Address: 0x631d50
//
__int64 __fastcall sub_631D50(__int64 a1)
{
  __int64 result; // rax
  char i; // dl

  result = sub_8D23E0(*(_QWORD *)(a1 + 120));
  if ( (_DWORD)result )
  {
    result = *(_QWORD *)(*(_QWORD *)(a1 + 184) + 128LL);
    for ( i = *(_BYTE *)(result + 140); i == 12; i = *(_BYTE *)(result + 140) )
      result = *(_QWORD *)(result + 160);
    if ( i == 8 && (*(_WORD *)(result + 168) & 0x180) == 0 )
      return sub_62F730((__int64 *)(a1 + 120), *(_QWORD *)(result + 176), *(_BYTE *)(result + 168) >> 7);
  }
  return result;
}
