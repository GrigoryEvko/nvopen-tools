// Function: sub_8D4050
// Address: 0x8d4050
//
__int64 __fastcall sub_8D4050(__int64 a1)
{
  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  return *(_QWORD *)(a1 + 160);
}
