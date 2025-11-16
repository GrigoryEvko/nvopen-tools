// Function: sub_255FB20
// Address: 0x255fb20
//
char __fastcall sub_255FB20(__int64 a1, _QWORD *a2)
{
  if ( *(_QWORD *)(a1 + 8) == a2[1] && *(_QWORD *)a1 == *a2 )
    return sub_254C7C0(*(__int64 **)(a1 + 16), a2[2]);
  else
    return 0;
}
