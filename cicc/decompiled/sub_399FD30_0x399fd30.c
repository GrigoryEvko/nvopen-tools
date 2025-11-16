// Function: sub_399FD30
// Address: 0x399fd30
//
void __fastcall sub_399FD30(__int64 a1)
{
  unsigned int v1; // esi
  unsigned int v2; // edx

  v1 = *(_DWORD *)(a1 + 68);
  if ( v1 )
  {
    v2 = *(_DWORD *)(a1 + 72);
    if ( v2 )
      sub_399EA60(a1, v1, v2);
  }
}
