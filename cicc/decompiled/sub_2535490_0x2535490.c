// Function: sub_2535490
// Address: 0x2535490
//
void __fastcall sub_2535490(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  if ( *(_QWORD *)(a1 + 8) <= a3 )
    a3 = *(_QWORD *)(a1 + 8);
  if ( *(_QWORD *)(a1 + 16) <= a2 )
    a2 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 16) = a2;
}
