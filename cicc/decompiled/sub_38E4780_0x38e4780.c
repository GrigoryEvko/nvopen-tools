// Function: sub_38E4780
// Address: 0x38e4780
//
__int64 __fastcall sub_38E4780(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 result; // rax
  int v5; // [rsp+Ch] [rbp-74h] BYREF
  _QWORD v6[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v7[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v8[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v9; // [rsp+40h] [rbp-40h]
  _QWORD v10[2]; // [rsp+50h] [rbp-30h] BYREF
  __int16 v11; // [rsp+60h] [rbp-20h]

  v6[0] = a2;
  v6[1] = a3;
  v5 = a4;
  v7[0] = a1;
  v7[1] = &v5;
  result = sub_3909F10(a1, sub_38EC400, v7, 1);
  if ( (_BYTE)result )
  {
    v8[0] = " in '";
    v8[1] = v6;
    v9 = 1283;
    v10[0] = v8;
    v10[1] = "' directive";
    v11 = 770;
    return sub_39094A0(a1, v10);
  }
  return result;
}
