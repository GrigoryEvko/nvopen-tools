// Function: sub_926800
// Address: 0x926800
//
__int64 __fastcall sub_926800(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v3; // al

  v3 = *(_BYTE *)(a3 + 24);
  if ( v3 == 3 )
  {
    sub_922820(a1, a2, a3);
    return a1;
  }
  else if ( v3 > 3u )
  {
    if ( v3 != 20 )
      goto LABEL_9;
    sub_9228E0(a1, a2, a3);
    return a1;
  }
  else
  {
    if ( v3 != 1 )
    {
      if ( v3 == 2 )
      {
        sub_922F70(a1, a2, (__int64 *)a3);
        return a1;
      }
LABEL_9:
      sub_91B8A0("cannot generate l-value for this expression!", (_DWORD *)(a3 + 36), 1);
    }
    sub_926620(a1, a2, a3);
    return a1;
  }
}
