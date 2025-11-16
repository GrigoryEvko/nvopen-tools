// Function: sub_102B9D0
// Address: 0x102b9d0
//
unsigned __int64 __fastcall sub_102B9D0(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  unsigned __int64 v3; // rsi

  result = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)(result + 8) == 14 )
  {
    v3 = a2 & 0xFFFFFFFFFFFFFFFBLL;
    sub_102B270(a1, v3);
    return sub_102B270(a1, v3 | 4);
  }
  return result;
}
