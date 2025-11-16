// Function: sub_D4B680
// Address: 0xd4b680
//
__int64 __fastcall sub_D4B680(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax

  v4 = sub_D4B410(a2, a3);
  if ( v4 )
    sub_D489E0(a1, a2, v4, a3);
  else
    *(_BYTE *)(a1 + 48) = 0;
  return a1;
}
