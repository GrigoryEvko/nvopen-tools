// Function: sub_3965950
// Address: 0x3965950
//
__int64 __fastcall sub_3965950(__int64 a1, __int64 *a2, __int64 *a3, __int64 *a4, __int64 a5)
{
  __int64 v9; // rsi

  if ( !(unsigned __int8)sub_3960EF0((_BYTE *)*a2) )
    return sub_3964ED0(a1, a2, a3, a4, a5);
  v9 = *a2;
  if ( *(_BYTE *)(*a2 + 16) != 17 )
    v9 = 0;
  return sub_39627B0(a1, v9, a5);
}
