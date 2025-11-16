// Function: sub_1210620
// Address: 0x1210620
//
__int64 __fastcall sub_1210620(__int64 *a1, _QWORD *a2)
{
  unsigned int v2; // r13d
  _QWORD *v3; // rax
  _QWORD *v4; // r13
  const char *v6; // rax
  unsigned __int64 v7; // rsi
  char v8; // [rsp+Bh] [rbp-55h] BYREF
  int v9; // [rsp+Ch] [rbp-54h] BYREF
  const char *v10; // [rsp+10h] [rbp-50h] BYREF
  char v11; // [rsp+30h] [rbp-30h]
  char v12; // [rsp+31h] [rbp-2Fh]

  v2 = 1;
  v9 = 0;
  v8 = 1;
  if ( (unsigned __int8)sub_120E4B0((__int64)a1, 1u, &v8, &v9) )
    return v2;
  if ( v9 == 1 )
  {
    v12 = 1;
    v6 = "fence cannot be unordered";
  }
  else
  {
    if ( v9 != 2 )
    {
      v3 = sub_BD2C40(80, unk_3F222C8);
      v4 = v3;
      if ( v3 )
        sub_B4D930((__int64)v3, *a1, v9, v8, 0, 0);
      *a2 = v4;
      return 0;
    }
    v12 = 1;
    v6 = "fence cannot be monotonic";
  }
  v7 = a1[29];
  v10 = v6;
  v11 = 3;
  sub_11FD800((__int64)(a1 + 22), v7, (__int64)&v10, 1);
  return 1;
}
