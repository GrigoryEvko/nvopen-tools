// Function: sub_9C3870
// Address: 0x9c3870
//
__int64 __fastcall sub_9C3870(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // rsi

  if ( *a1 )
    result = sub_B91220(a1);
  v8 = *a2;
  *a1 = *a2;
  if ( v8 )
  {
    result = sub_B976B0(a2, v8, a1, a4, a5, a6);
    *a2 = 0;
  }
  return result;
}
