// Function: sub_64FF30
// Address: 0x64ff30
//
__int64 __fastcall sub_64FF30(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  __int64 v3; // rsi

  v1 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 && *(_BYTE *)(v1 + 80) == 7 && (*(_BYTE *)(*(_QWORD *)(v1 + 88) + 176LL) & 8) != 0 )
  {
    sub_6851C0(2501, a1 + 260);
  }
  else if ( *(_BYTE *)(a1 + 268) )
  {
    sub_6851C0(80, a1 + 260);
  }
  if ( (*(_BYTE *)(a1 + 8) & 0x20) != 0 )
    sub_6851C0(255, a1 + 32);
  result = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 && *(_BYTE *)(result + 80) == 7 )
  {
    v3 = *(_QWORD *)(result + 88);
    if ( (*(_BYTE *)(v3 + 156) & 2) != 0 )
      return sub_6851C0(3649, v3 + 64);
  }
  return result;
}
