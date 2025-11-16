// Function: sub_3227110
// Address: 0x3227110
//
__int64 __fastcall sub_3227110(__int64 a1)
{
  __int64 v1; // rsi
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 3056);
  if ( v1 )
    result = sub_3226DF0(a1, v1);
  *(_QWORD *)(a1 + 3056) = 0;
  *(_QWORD *)(a1 + 3048) = 0;
  return result;
}
