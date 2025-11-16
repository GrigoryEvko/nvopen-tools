// Function: sub_31BC080
// Address: 0x31bc080
//
bool __fastcall sub_31BC080(_QWORD *a1, _QWORD *a2)
{
  if ( !*a2 )
    return 1;
  if ( *a1 && !sub_B445A0(*(_QWORD *)(a2[1] + 16LL), *(_QWORD *)(*a1 + 16LL)) )
    return sub_B445A0(*(_QWORD *)(a1[1] + 16LL), *(_QWORD *)(*a2 + 16LL));
  return 1;
}
