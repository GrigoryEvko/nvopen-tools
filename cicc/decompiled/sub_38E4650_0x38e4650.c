// Function: sub_38E4650
// Address: 0x38e4650
//
__int64 __fastcall sub_38E4650(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  _QWORD v4[3]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v5; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v6[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v7; // [rsp+30h] [rbp-40h]
  _QWORD v8[2]; // [rsp+40h] [rbp-30h] BYREF
  __int16 v9; // [rsp+50h] [rbp-20h]

  v4[0] = a2;
  v4[1] = a3;
  v5 = a1;
  result = sub_3909F10(a1, sub_38EE520, &v5, 1);
  if ( (_BYTE)result )
  {
    v6[0] = " in '";
    v6[1] = v4;
    v7 = 1283;
    v8[0] = v6;
    v8[1] = "' directive";
    v9 = 770;
    return sub_39094A0(a1, v8);
  }
  return result;
}
