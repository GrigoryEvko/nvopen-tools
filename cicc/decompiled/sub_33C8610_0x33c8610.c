// Function: sub_33C8610
// Address: 0x33c8610
//
bool __fastcall sub_33C8610(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rbx

  v2 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  if ( *(void **)(v2 + 24) == sub_C33340() )
    v3 = *(_QWORD *)(v2 + 32);
  else
    v3 = v2 + 24;
  return (*(_BYTE *)(v3 + 20) & 7) != 3;
}
