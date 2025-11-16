// Function: sub_38E4820
// Address: 0x38e4820
//
__int64 __fastcall sub_38E4820(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  _QWORD v5[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v6[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v7[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v8; // [rsp+30h] [rbp-40h]
  _QWORD v9[2]; // [rsp+40h] [rbp-30h] BYREF
  __int16 v10; // [rsp+50h] [rbp-20h]

  v5[0] = a2;
  v5[1] = a3;
  v6[1] = a4;
  v6[0] = a1;
  result = sub_3909F10(a1, sub_38F5CA0, v6, 1);
  if ( (_BYTE)result )
  {
    v7[0] = " in '";
    v7[1] = v5;
    v8 = 1283;
    v9[0] = v7;
    v9[1] = "' directive";
    v10 = 770;
    return sub_39094A0(a1, v9);
  }
  return result;
}
