// Function: sub_173D840
// Address: 0x173d840
//
_BOOL8 __fastcall sub_173D840(__int64 a1)
{
  __int64 v1; // rbx

  v1 = a1;
  if ( *(void **)(a1 + 8) == sub_16982C0() )
    v1 = *(_QWORD *)(a1 + 16);
  return (*(_BYTE *)(v1 + 26) & 8) != 0;
}
