// Function: sub_5C6E50
// Address: 0x5c6e50
//
_QWORD *__fastcall sub_5C6E50(__int64 a1, _QWORD *a2, char a3)
{
  __int64 v3; // rax

  v3 = *(_QWORD *)(a1 + 48);
  if ( unk_4F077BC && !unk_4F077B4 )
  {
    sub_684B30(2769, a1 + 56);
    *(_BYTE *)(a1 + 8) = 0;
    return a2;
  }
  else if ( a3 == 3 )
  {
    sub_643E40(sub_5CDC30, *(_QWORD *)(v3 + 424), 0);
    return a2;
  }
  else
  {
    if ( a3 != 11 )
      sub_721090(a1);
    if ( v3 && (*(_BYTE *)(v3 + 127) & 0x10) == 0 && a1 == sub_736C60(5, a2[13]) )
    {
      sub_6854C0(1855, a1 + 56, *a2);
      *(_BYTE *)(a1 + 8) = 0;
    }
    return a2;
  }
}
