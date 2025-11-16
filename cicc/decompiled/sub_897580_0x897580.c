// Function: sub_897580
// Address: 0x897580
//
__int64 __fastcall sub_897580(__int64 a1, __int64 *a2, __int64 a3)
{
  _QWORD *v4; // rdi
  __int64 result; // rax
  __int64 v6; // [rsp-20h] [rbp-20h]

  if ( a2 )
  {
    v4 = *(_QWORD **)(a1 + 336);
    if ( !*v4 )
    {
      v6 = a3;
      result = sub_877D80((__int64)v4, a2);
      a3 = v6;
    }
    if ( !*(_QWORD *)(a3 + 104) )
    {
      result = *(_QWORD *)(a1 + 336);
      *(_QWORD *)(a3 + 104) = result;
    }
  }
  return result;
}
