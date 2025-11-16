// Function: sub_1442240
// Address: 0x1442240
//
char __fastcall sub_1442240(__int64 a1, _QWORD *a2, __int64 *a3)
{
  char result; // al
  unsigned __int64 v4; // [rsp+0h] [rbp-20h] BYREF
  char v5; // [rsp+8h] [rbp-18h]

  sub_1441B50((__int64)&v4, a1, *a2 & 0xFFFFFFFFFFFFFFF8LL, a3);
  result = v5;
  if ( v5 )
    return sub_1441CD0(a1, v4);
  return result;
}
