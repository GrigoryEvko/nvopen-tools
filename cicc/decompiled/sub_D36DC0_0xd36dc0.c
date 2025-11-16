// Function: sub_D36DC0
// Address: 0xd36dc0
//
__int64 __fastcall sub_D36DC0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v4; // rax

  v2 = *(_QWORD *)(*a1 + 112LL);
  if ( !(unsigned __int8)sub_D97040(v2, *(_QWORD *)(a2 + 8)) )
    return 0;
  v4 = sub_DD8400(v2, a2);
  return sub_DADE90(v2, v4, a1[3]);
}
