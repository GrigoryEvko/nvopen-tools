// Function: sub_B2DAA0
// Address: 0xb2daa0
//
__int64 __fastcall sub_B2DAA0(__int64 a1)
{
  unsigned int v1; // ebx
  __int64 v2; // rax
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  __int64 v5; // rsi
  unsigned __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // r14
  char v9; // al
  char v10; // r12
  char v12; // [rsp+7h] [rbp-39h] BYREF
  __int64 v13; // [rsp+8h] [rbp-38h] BYREF
  __int64 v14; // [rsp+10h] [rbp-30h] BYREF
  unsigned __int64 v15; // [rsp+18h] [rbp-28h]

  v1 = -1;
  v13 = sub_B2D7E0(a1, "denormal-fp-math-f32", 0x14u);
  if ( v13 )
  {
    v2 = sub_A72240(&v13);
    v15 = v3;
    v14 = v2;
    v12 = 44;
    v4 = sub_C931B0(&v14, &v12, 1, 0);
    if ( v4 == -1 )
    {
      LOBYTE(v1) = sub_B2B6F0(v14, v15);
      BYTE1(v1) = v1;
    }
    else
    {
      v5 = v15;
      v6 = v4 + 1;
      if ( v4 + 1 > v15 )
      {
        if ( v4 <= v15 )
          v5 = v4;
        LOBYTE(v1) = sub_B2B6F0(v14, v5);
        BYTE1(v1) = v1;
      }
      else
      {
        v7 = v14 + v6;
        v8 = v15 - v6;
        if ( v4 <= v15 )
          v5 = v4;
        v9 = sub_B2B6F0(v14, v5);
        v10 = v9;
        if ( v8 )
          v9 = sub_B2B6F0(v7, v8);
        LOBYTE(v1) = v10;
        BYTE1(v1) = v9;
      }
    }
  }
  return v1;
}
