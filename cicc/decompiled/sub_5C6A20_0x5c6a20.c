// Function: sub_5C6A20
// Address: 0x5c6a20
//
__int64 __fastcall sub_5C6A20(__int64 a1)
{
  __int64 result; // rax

  if ( unk_4F06418 == 118 )
  {
    *(_QWORD *)(a1 + 16) = sub_724840(unk_4F073B8, "restrict");
  }
  else if ( *(_BYTE *)(a1 + 9) == 4 && unk_4F077C4 != 2 )
  {
    *(_QWORD *)(a1 + 16) = sub_724840(unk_4F073B8, "_Alignas");
  }
  else
  {
    *(_QWORD *)(a1 + 16) = sub_7C9D40();
  }
  result = unk_4F063F0;
  *(_QWORD *)(a1 + 64) = unk_4F063F0;
  return result;
}
