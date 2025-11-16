// Function: sub_CE02D0
// Address: 0xce02d0
//
__int64 __fastcall sub_CE02D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int128 *v6; // rax

  if ( !a2 )
    return 0;
  sub_CE00D0(a1);
  sub_CE0220(a1, a2, v2, v3, v4);
  if ( !*(_DWORD *)(a1 + 72) )
  {
    v6 = sub_BC2B00();
    sub_F11020(v6);
  }
  return 0;
}
