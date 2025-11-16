// Function: sub_399FA80
// Address: 0x399fa80
//
__int64 __fastcall sub_399FA80(__int64 a1)
{
  unsigned int v1; // esi

  v1 = *(_DWORD *)(a1 + 72);
  if ( v1 )
    sub_399EAF0(a1, v1);
  return sub_399EB30(a1, (1 << *(_DWORD *)(a1 + 68)) - 1);
}
