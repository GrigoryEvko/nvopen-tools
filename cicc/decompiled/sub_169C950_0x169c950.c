// Function: sub_169C950
// Address: 0x169c950
//
_BOOL8 __fastcall sub_169C950(__int64 a1)
{
  __int64 v1; // rbx

  v1 = *(_QWORD *)(a1 + 8);
  if ( *(void **)(v1 + 8) == sub_16982C0() )
    v1 = *(_QWORD *)(v1 + 16);
  return (*(_BYTE *)(v1 + 26) & 8) != 0;
}
