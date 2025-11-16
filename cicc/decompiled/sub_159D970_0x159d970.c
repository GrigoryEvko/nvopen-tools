// Function: sub_159D970
// Address: 0x159d970
//
__int64 __fastcall sub_159D970(__int64 a1)
{
  __int64 v3; // rdi
  __int64 v4; // rdi

  if ( *(_BYTE *)(a1 + 16) <= 3u )
    return 0;
  while ( 1 )
  {
    v3 = *(_QWORD *)(a1 + 8);
    if ( !v3 )
      break;
    v4 = sub_1648700(v3);
    if ( *(_BYTE *)(v4 + 16) > 0x10u || !(unsigned __int8)sub_159D970(v4) )
      return 0;
  }
  sub_159D850(a1);
  return 1;
}
