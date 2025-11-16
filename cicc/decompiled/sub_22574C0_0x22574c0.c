// Function: sub_22574C0
// Address: 0x22574c0
//
__int64 __fastcall sub_22574C0(__int64 a1, __int64 a2, int a3)
{
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  if ( a3 == 1 )
    sub_2241130((unsigned __int64 *)a1, 0, 0, "iostream error", 0xEu);
  else
    sub_2241130((unsigned __int64 *)a1, 0, 0, "Unknown error", 0xDu);
  return a1;
}
