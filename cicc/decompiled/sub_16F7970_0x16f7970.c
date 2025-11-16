// Function: sub_16F7970
// Address: 0x16f7970
//
__int64 __fastcall sub_16F7970(__int64 a1)
{
  _BYTE *v1; // rax

  v1 = sub_16F7720(a1, *(_BYTE **)(a1 + 40));
  if ( *(_BYTE **)(a1 + 40) == v1 )
    return 0;
  *(_QWORD *)(a1 + 40) = v1;
  ++*(_DWORD *)(a1 + 64);
  *(_DWORD *)(a1 + 60) = 0;
  return 1;
}
