// Function: sub_6E9820
// Address: 0x6e9820
//
__int64 __fastcall sub_6E9820(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rax

  v2 = *(_BYTE *)(a1 + 16);
  if ( v2 == 1 )
  {
    v4 = *(_QWORD *)(a1 + 144);
    if ( *(_BYTE *)(v4 + 24) != 2 )
      return 0;
    return (unsigned int)sub_711520(*(_QWORD *)(v4 + 56), a2) != 0;
  }
  else
  {
    if ( v2 != 2 )
      return 0;
    return sub_711520(a1 + 144, a2);
  }
}
