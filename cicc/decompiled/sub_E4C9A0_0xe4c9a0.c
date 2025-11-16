// Function: sub_E4C9A0
// Address: 0xe4c9a0
//
__int64 __fastcall sub_E4C9A0(__int64 a1, signed __int64 a2)
{
  __int64 v3; // [rsp+8h] [rbp-18h]

  if ( !*(_BYTE *)(*(_QWORD *)(a1 + 312) + 351LL)
    && (v3 = sub_E92200(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 160LL), a2, 1), BYTE4(v3)) )
  {
    return (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 320) + 40LL))(
             *(_QWORD *)(a1 + 320),
             *(_QWORD *)(a1 + 304),
             (unsigned int)v3);
  }
  else
  {
    return sub_CB59F0(*(_QWORD *)(a1 + 304), a2);
  }
}
