// Function: sub_F131D0
// Address: 0xf131d0
//
__int64 __fastcall sub_F131D0(unsigned __int8 *a1, unsigned __int8 *a2)
{
  __int64 result; // rax
  char v3; // [rsp+Dh] [rbp-73h] BYREF
  char v4; // [rsp+Eh] [rbp-72h] BYREF
  char v5; // [rsp+Fh] [rbp-71h] BYREF
  _QWORD v6[2]; // [rsp+10h] [rbp-70h] BYREF
  unsigned __int8 *v7; // [rsp+20h] [rbp-60h]
  unsigned __int8 *v8[10]; // [rsp+30h] [rbp-50h] BYREF

  v8[2] = a1;
  v6[0] = &v4;
  v8[1] = (unsigned __int8 *)&v3;
  v8[3] = (unsigned __int8 *)&v4;
  v6[1] = &v5;
  v7 = (unsigned __int8 *)sub_B43CC0((__int64)a2);
  v8[0] = a2;
  v8[4] = (unsigned __int8 *)v6;
  v8[5] = (unsigned __int8 *)&v5;
  v8[6] = v7;
  result = sub_F12970(v8, 0);
  if ( !result )
    return sub_F12970(v8, 1);
  return result;
}
