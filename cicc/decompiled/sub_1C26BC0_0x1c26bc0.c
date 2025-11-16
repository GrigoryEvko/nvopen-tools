// Function: sub_1C26BC0
// Address: 0x1c26bc0
//
__int64 __fastcall sub_1C26BC0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v6; // rax

  if ( !a2 )
    return 0;
  sub_1C268F0(a1);
  sub_1C26A00(a1, a2, v2, v3, v4);
  if ( !*(_DWORD *)(a1 + 72) )
  {
    v6 = sub_163A1D0();
    sub_1705E40(v6);
  }
  return 0;
}
