// Function: sub_2535A50
// Address: 0x2535a50
//
__int64 __fastcall sub_2535A50(__int64 a1)
{
  int v2; // eax

  if ( !*(_DWORD *)(a1 + 8) )
    return 0;
  LOBYTE(v2) = sub_AAF760(a1 + 16);
  return v2 ^ 1u;
}
