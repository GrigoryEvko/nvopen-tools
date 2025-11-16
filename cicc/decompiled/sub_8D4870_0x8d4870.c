// Function: sub_8D4870
// Address: 0x8d4870
//
__int64 __fastcall sub_8D4870(__int64 a1)
{
  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  return *(_QWORD *)(a1 + 168);
}
