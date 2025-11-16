// Function: sub_17C5B00
// Address: 0x17c5b00
//
__int64 __fastcall sub_17C5B00(_QWORD *a1)
{
  __int64 v1; // rdx
  __int64 v2; // rsi
  __int64 result; // rax

  v1 = a1[19];
  v2 = a1[18];
  if ( v1 != v2 )
    return sub_1B28040(a1[5], v2, (v1 - v2) >> 3);
  return result;
}
