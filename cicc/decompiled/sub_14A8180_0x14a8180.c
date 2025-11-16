// Function: sub_14A8180
// Address: 0x14a8180
//
__int64 __fastcall sub_14A8180(__int64 a1, __int64 *a2, char a3)
{
  __int64 v4; // rsi
  __int64 v5; // rsi
  __int64 v6; // rsi
  __int64 result; // rax
  __int64 v8; // rax

  v4 = *(_QWORD *)(a1 + 48);
  if ( a3 )
  {
    if ( v4 || *(__int16 *)(a1 + 18) < 0 )
      v4 = sub_1625790(a1, 1);
    *a2 = sub_14A8140(*a2, v4);
    v5 = *(_QWORD *)(a1 + 48);
    if ( v5 || *(__int16 *)(a1 + 18) < 0 )
      v5 = sub_1625790(a1, 7);
    a2[1] = sub_1631A90(a2[1], v5);
    v6 = *(_QWORD *)(a1 + 48);
    if ( v6 || *(__int16 *)(a1 + 18) < 0 )
      v6 = sub_1625790(a1, 8);
    result = sub_1630FC0(a2[2], v6);
    a2[2] = result;
  }
  else
  {
    if ( v4 || *(__int16 *)(a1 + 18) < 0 )
      v4 = sub_1625790(a1, 1);
    *a2 = v4;
    v8 = *(_QWORD *)(a1 + 48);
    if ( v8 || *(__int16 *)(a1 + 18) < 0 )
      v8 = sub_1625790(a1, 7);
    a2[1] = v8;
    result = *(_QWORD *)(a1 + 48);
    if ( result || *(__int16 *)(a1 + 18) < 0 )
      result = sub_1625790(a1, 8);
    a2[2] = result;
  }
  return result;
}
