// Function: sub_2AA7E40
// Address: 0x2aa7e40
//
__int64 __fastcall sub_2AA7E40(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3[3]; // [rsp+0h] [rbp-20h] BYREF

  result = sub_DFB220(a2);
  v3[1] = result;
  if ( BYTE4(result) )
    goto LABEL_2;
  if ( (unsigned __int8)sub_B2D610(a1, 96) )
  {
    v3[0] = sub_B2D7D0(a1, 96);
    result = sub_A71ED0(v3);
LABEL_2:
    v3[0] = result;
    return result;
  }
  BYTE4(v3[0]) = 0;
  return v3[0];
}
