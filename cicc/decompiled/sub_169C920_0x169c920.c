// Function: sub_169C920
// Address: 0x169c920
//
__int64 __fastcall sub_169C920(__int64 a1)
{
  __int64 v1; // rbx

  v1 = *(_QWORD *)(a1 + 8);
  if ( *(void **)(v1 + 8) == sub_16982C0() )
    v1 = *(_QWORD *)(v1 + 16);
  return *(_BYTE *)(v1 + 26) & 7;
}
