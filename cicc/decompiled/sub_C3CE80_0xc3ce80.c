// Function: sub_C3CE80
// Address: 0xc3ce80
//
_BOOL8 __fastcall sub_C3CE80(__int64 a1)
{
  __int64 v1; // rbx

  v1 = *(_QWORD *)(a1 + 8);
  if ( *(void **)v1 == sub_C33340() )
    v1 = *(_QWORD *)(v1 + 8);
  return (*(_BYTE *)(v1 + 20) & 8) != 0;
}
