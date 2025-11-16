// Function: sub_20A1280
// Address: 0x20a1280
//
char *__fastcall sub_20A1280(__int64 a1, __int64 a2, __int64 a3)
{
  bool v3; // al
  bool v4; // zf
  char *result; // rax
  bool v6; // r8
  _QWORD v7[3]; // [rsp+0h] [rbp-20h] BYREF

  v7[0] = a2;
  v7[1] = a3;
  if ( (_BYTE)a2 )
  {
    if ( (unsigned __int8)(a2 - 2) <= 5u || (unsigned __int8)(a2 - 14) <= 0x47u )
      return "r";
    v3 = (unsigned __int8)(a2 - 8) <= 5u || (unsigned __int8)(a2 - 86) <= 0x17u;
  }
  else
  {
    v6 = sub_1F58CF0((__int64)v7);
    result = "r";
    if ( v6 )
      return result;
    v3 = sub_1F58CD0((__int64)v7);
  }
  v4 = !v3;
  result = "f";
  if ( v4 )
    return 0;
  return result;
}
