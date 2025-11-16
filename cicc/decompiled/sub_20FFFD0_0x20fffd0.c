// Function: sub_20FFFD0
// Address: 0x20fffd0
//
bool __fastcall sub_20FFFD0(__int64 a1, __int64 a2)
{
  if ( !*(_BYTE *)(a1 + 68) )
    sub_20FFD80(a1, a2);
  return *(_DWORD *)(a1 + 108) != *(_DWORD *)(a1 + 112);
}
