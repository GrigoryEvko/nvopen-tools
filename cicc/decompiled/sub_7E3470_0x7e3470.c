// Function: sub_7E3470
// Address: 0x7e3470
//
__int64 __fastcall sub_7E3470(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  if ( !a2 )
    return ~*(_QWORD *)(*(_QWORD *)(a1 + 168) + 56LL);
  v2 = sub_7E0220(a2);
  if ( v2 )
    return ~*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v2 + 40) + 168LL) + 48LL);
  else
    return ~*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 168LL) + 56LL);
}
