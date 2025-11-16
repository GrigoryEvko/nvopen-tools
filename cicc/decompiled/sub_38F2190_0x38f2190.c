// Function: sub_38F2190
// Address: 0x38f2190
//
__int64 __fastcall sub_38F2190(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  char v4; // bl
  unsigned __int8 v5; // al
  __int64 result; // rax
  _QWORD v7[2]; // [rsp+0h] [rbp-80h] BYREF
  _BYTE *v8; // [rsp+10h] [rbp-70h] BYREF
  __int64 v9; // [rsp+18h] [rbp-68h]
  _QWORD v10[2]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v11; // [rsp+30h] [rbp-50h]
  _QWORD v12[2]; // [rsp+40h] [rbp-40h] BYREF
  __int16 v13; // [rsp+50h] [rbp-30h]

  v4 = a4;
  v7[0] = a2;
  v7[1] = a3;
  v8 = 0;
  v9 = 0;
  v10[0] = "expected identifier";
  v11 = 259;
  v5 = sub_38F0EE0(a1, (__int64 *)&v8, a3, a4);
  if ( (unsigned __int8)sub_3909CB0(a1, v5, v10)
    || (v13 = 259, v12[0] = "unexpected token", (unsigned __int8)sub_3909E20(a1, 25, v12))
    || (result = sub_38E8800(a1, v8, v9, v4, 1), (_BYTE)result) )
  {
    v12[0] = v10;
    v13 = 770;
    v10[0] = " in '";
    v10[1] = v7;
    v11 = 1283;
    v12[1] = "' directive";
    return sub_39094A0(a1, v12);
  }
  return result;
}
