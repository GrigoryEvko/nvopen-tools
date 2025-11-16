// Function: sub_9380F0
// Address: 0x9380f0
//
__int64 __fastcall sub_9380F0(_QWORD **a1, __int64 a2, char a3)
{
  while ( *(_BYTE *)(a2 + 140) == 12 )
    a2 = *(_QWORD *)(a2 + 160);
  return sub_937D20(a1, *(_QWORD *)(a2 + 160), **(__int64 ***)(a2 + 168), a3);
}
