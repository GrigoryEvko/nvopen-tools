// Function: sub_1403A20
// Address: 0x1403a20
//
__int64 __fastcall sub_1403A20(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rdx

  v2 = sub_1403960(*(_QWORD **)(a2 + 32), *(_QWORD *)(a2 + 40));
  if ( v3 == v2 )
    return 0;
  v4 = sub_1649960(*(_QWORD *)(*v2 + 56LL));
  if ( !(unsigned __int8)sub_160E740(v4, v5) )
    return 0;
  sub_13FC6E0(a2, *(_QWORD *)(a1 + 160), a1 + 168);
  return 0;
}
