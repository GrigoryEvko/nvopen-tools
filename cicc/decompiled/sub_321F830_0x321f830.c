// Function: sub_321F830
// Address: 0x321f830
//
__int64 __fastcall sub_321F830(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax

  v1 = a1 + 3776;
  if ( !*(_BYTE *)(a1 + 3769) )
    v1 = a1 + 3080;
  v2 = sub_31DA6B0(*(_QWORD *)(a1 + 8));
  return sub_3245010(v1, *(_QWORD *)(v2 + 80));
}
