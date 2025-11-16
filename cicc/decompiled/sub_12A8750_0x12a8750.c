// Function: sub_12A8750
// Address: 0x12a8750
//
__int64 __fastcall sub_12A8750(__int64 a1, unsigned int a2, __int64 *a3, __int64 *a4)
{
  __int64 v6; // r12
  __int64 v7; // r15
  __int64 result; // rax

  v6 = sub_16432A0(*(_QWORD *)(a1 + 360));
  v7 = sub_16463B0(v6, 2);
  result = sub_16463B0(v6, 4);
  if ( a2 <= 0x1CD )
  {
    if ( a2 > 0x1CA )
    {
      *a3 = v7;
      result = sub_1646BA0(v7, 0);
      *a4 = result;
    }
  }
  else if ( a2 - 466 <= 2 )
  {
    *a3 = result;
    result = sub_1646BA0(result, 0);
    *a4 = result;
  }
  return result;
}
