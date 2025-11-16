// Function: sub_11FA890
// Address: 0x11fa890
//
__int64 __fastcall sub_11FA890(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  const char *v6; // rax
  void *v7; // rdx
  __int64 *v9; // [rsp+8h] [rbp-B8h] BYREF
  void *v10[4]; // [rsp+10h] [rbp-B0h] BYREF
  __int16 v11; // [rsp+30h] [rbp-90h]
  void *v12; // [rsp+40h] [rbp-80h] BYREF
  __int16 v13; // [rsp+60h] [rbp-60h]
  _BYTE v14[80]; // [rsp+70h] [rbp-50h] BYREF

  sub_11F3840((__int64)v14, a1, a2, a3, a4);
  v14[40] = byte_4F91E88;
  v14[41] = qword_4F91CC8;
  v14[42] = byte_4F91DA8;
  v13 = 257;
  v6 = sub_BD5D20(a1);
  v11 = 1283;
  v10[0] = "cfg.";
  v10[3] = v7;
  v10[2] = (void *)v6;
  v9 = (__int64 *)v14;
  sub_11FA7D0(&v9, v10, a5, &v12, 0);
  return sub_11F3870((__int64)v14);
}
