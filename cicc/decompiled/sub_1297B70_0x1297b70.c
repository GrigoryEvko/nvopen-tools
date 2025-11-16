// Function: sub_1297B70
// Address: 0x1297b70
//
__int64 __fastcall sub_1297B70(_QWORD **a1, __int64 a2, char a3)
{
  while ( *(_BYTE *)(a2 + 140) == 12 )
    a2 = *(_QWORD *)(a2 + 160);
  return sub_12977E0(a1, *(_QWORD *)(a2 + 160), **(__int64 ***)(a2 + 168), a3);
}
