// Function: sub_1049690
// Address: 0x1049690
//
__int64 __fastcall sub_1049690(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 result; // rax

  *a1 = a2;
  a1[1] = 0;
  a1[2] = 0;
  v2 = sub_B2BE50(a2);
  result = sub_B6E900(v2);
  if ( (_BYTE)result )
    return sub_1048E60((__int64)a1, a2);
  return result;
}
