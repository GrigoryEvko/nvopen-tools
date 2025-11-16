// Function: sub_38EB0A0
// Address: 0x38eb0a0
//
__int64 __fastcall sub_38EB0A0(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // rdx
  __int64 result; // rax
  __int64 v4; // [rsp+0h] [rbp-70h] BYREF
  __int64 v5; // [rsp+8h] [rbp-68h]
  _QWORD v6[2]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v7; // [rsp+20h] [rbp-50h]
  _QWORD v8[2]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v9; // [rsp+40h] [rbp-30h]

  v1 = sub_3909290(a1 + 144);
  v4 = sub_38EAF10(a1);
  v5 = v2;
  v8[0] = "unexpected token in '.abort' directive";
  v9 = 259;
  result = sub_3909E20(a1, 9, v8);
  if ( !(_BYTE)result )
  {
    if ( v5 )
    {
      v7 = 1283;
      v6[0] = ".abort '";
      v6[1] = &v4;
      v8[0] = v6;
      v8[1] = "' detected. Assembly stopping.";
      v9 = 770;
    }
    else
    {
      v8[0] = ".abort detected. Assembly stopping.";
      v9 = 259;
    }
    return sub_3909790(a1, v1, v8, 0, 0);
  }
  return result;
}
