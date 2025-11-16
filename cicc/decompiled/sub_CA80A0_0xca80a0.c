// Function: sub_CA80A0
// Address: 0xca80a0
//
__int64 __fastcall sub_CA80A0(__int64 a1)
{
  _BYTE *v1; // rax

  v1 = sub_CA7C80(a1, *(_BYTE **)(a1 + 40));
  if ( *(_BYTE **)(a1 + 40) == v1 )
    return 0;
  *(_QWORD *)(a1 + 40) = v1;
  ++*(_DWORD *)(a1 + 64);
  *(_DWORD *)(a1 + 60) = 0;
  return 1;
}
