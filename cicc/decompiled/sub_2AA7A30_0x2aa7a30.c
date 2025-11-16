// Function: sub_2AA7A30
// Address: 0x2aa7a30
//
__int64 __fastcall sub_2AA7A30(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx

  if ( a1[1] == a2 )
    return sub_31A6C30(*(_QWORD *)(*a1 + 40LL), a2);
  v2 = sub_AA5930(a2);
  if ( v2 == v3 )
    return sub_31A6C30(*(_QWORD *)(*a1 + 40LL), a2);
  sub_31A6C30(*(_QWORD *)(*a1 + 40LL), a2);
  return 1;
}
