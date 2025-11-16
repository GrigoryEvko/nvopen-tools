// Function: sub_645120
// Address: 0x645120
//
_QWORD *__fastcall sub_645120(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *result; // rax

  if ( *(_QWORD *)(a1 + 200) || *(_QWORD *)(a1 + 184) )
  {
    sub_6447A0(a1);
    sub_5CF700(*(__int64 **)(a1 + 200));
    sub_5CEC90(*(_QWORD **)(a1 + 200), a2, 3);
    sub_5CF700(*(__int64 **)(a1 + 184));
    sub_5CEC90(*(_QWORD **)(a1 + 184), a2, 3);
    v2 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a1 + 200) = 0;
    *(_QWORD *)(a1 + 288) = v2;
    *(_QWORD *)(a1 + 184) = 0;
    return sub_6447E0(a1);
  }
  return result;
}
