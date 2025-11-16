// Function: sub_2D2BFE0
// Address: 0x2d2bfe0
//
__int64 __fastcall sub_2D2BFE0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // r14
  __int64 v6; // r14

  if ( (_QWORD *)a2 == a1 + 1 )
  {
    if ( a1[5] && sub_2A4D650(a1[4] + 32LL, a3) )
      return 0;
    return sub_2D2BEA0((__int64)a1, a3);
  }
  if ( sub_2A4D650(a3, a2 + 32) )
  {
    result = a1[3];
    if ( result == a2 )
      return result;
    v5 = sub_220EF80(a2);
    if ( sub_2A4D650(v5 + 32, a3) )
    {
      result = 0;
      if ( *(_QWORD *)(v5 + 24) )
        return a2;
      return result;
    }
    return sub_2D2BEA0((__int64)a1, a3);
  }
  if ( !sub_2A4D650(a2 + 32, a3) )
    return a2;
  result = 0;
  if ( a1[4] != a2 )
  {
    v6 = sub_220EEE0(a2);
    if ( !sub_2A4D650(a3, v6 + 32) )
      return sub_2D2BEA0((__int64)a1, a3);
    result = 0;
    if ( *(_QWORD *)(a2 + 24) )
      return v6;
  }
  return result;
}
