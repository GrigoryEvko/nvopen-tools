// Function: sub_B2D9D0
// Address: 0xb2d9d0
//
__int64 __fastcall sub_B2D9D0(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rax
  __int64 v4; // rsi
  unsigned __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // r13
  unsigned __int8 v8; // al
  unsigned __int8 v9; // bl
  unsigned int v10; // edx
  char v12; // [rsp+7h] [rbp-39h] BYREF
  __int64 v13; // [rsp+8h] [rbp-38h] BYREF
  __int64 v14; // [rsp+10h] [rbp-30h] BYREF
  unsigned __int64 v15; // [rsp+18h] [rbp-28h]

  v13 = sub_B2D7E0(a1, "denormal-fp-math", 0x10u);
  v1 = sub_A72240(&v13);
  v15 = v2;
  v14 = v1;
  v12 = 44;
  v3 = sub_C931B0(&v14, &v12, 1, 0);
  if ( v3 == -1 )
  {
    v8 = sub_B2B6F0(v14, v15);
    v9 = v8;
  }
  else
  {
    v4 = v15;
    v5 = v3 + 1;
    if ( v3 + 1 > v15 )
    {
      if ( v3 <= v15 )
        v4 = v3;
      v8 = sub_B2B6F0(v14, v4);
      v9 = v8;
    }
    else
    {
      v6 = v14 + v5;
      v7 = v15 - v5;
      if ( v3 <= v15 )
        v4 = v3;
      v8 = sub_B2B6F0(v14, v4);
      v9 = v8;
      if ( v7 )
        v8 = sub_B2B6F0(v6, v7);
    }
  }
  v10 = v9;
  BYTE1(v10) = v8;
  return v10;
}
