// Function: sub_CC7380
// Address: 0xcc7380
//
__int64 __fastcall sub_CC7380(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  unsigned __int64 v3; // rcx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v13; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v14; // [rsp+8h] [rbp-28h]
  __int64 v15; // [rsp+10h] [rbp-20h] BYREF
  unsigned __int64 v16; // [rsp+18h] [rbp-18h]

  v1 = *a1;
  LOBYTE(v13) = 45;
  v15 = v1;
  v16 = a1[1];
  v2 = sub_C931B0(&v15, &v13, 1u, 0);
  if ( v2 == -1 )
  {
    v5 = 0;
    v6 = 0;
  }
  else
  {
    v3 = v16;
    v4 = v2 + 1;
    v5 = 0;
    if ( v4 <= v16 )
    {
      v3 = v4;
      v5 = v16 - v4;
    }
    v6 = v15 + v3;
  }
  v14 = v5;
  v13 = v6;
  LOBYTE(v15) = 45;
  v7 = sub_C931B0(&v13, &v15, 1u, 0);
  if ( v7 == -1 )
  {
    v10 = 0;
    v11 = 0;
  }
  else
  {
    v8 = v14;
    v9 = v7 + 1;
    v10 = 0;
    if ( v9 <= v14 )
    {
      v10 = v14 - v9;
      v8 = v9;
    }
    v11 = v13 + v8;
  }
  v14 = v10;
  v13 = v11;
  LOBYTE(v15) = 45;
  sub_C931B0(&v13, &v15, 1u, 0);
  return v13;
}
