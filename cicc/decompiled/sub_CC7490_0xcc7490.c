// Function: sub_CC7490
// Address: 0xcc7490
//
__int64 __fastcall sub_CC7490(__int64 *a1)
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
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rax
  __int64 v16; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v17; // [rsp+8h] [rbp-28h]
  __int64 v18; // [rsp+10h] [rbp-20h] BYREF
  unsigned __int64 v19; // [rsp+18h] [rbp-18h]

  v1 = *a1;
  LOBYTE(v16) = 45;
  v18 = v1;
  v19 = a1[1];
  v2 = sub_C931B0(&v18, &v16, 1u, 0);
  if ( v2 == -1 )
  {
    v5 = 0;
    v6 = 0;
  }
  else
  {
    v3 = v19;
    v4 = v2 + 1;
    v5 = 0;
    if ( v4 <= v19 )
    {
      v3 = v4;
      v5 = v19 - v4;
    }
    v6 = v18 + v3;
  }
  v17 = v5;
  v16 = v6;
  LOBYTE(v18) = 45;
  v7 = sub_C931B0(&v16, &v18, 1u, 0);
  if ( v7 == -1 )
  {
    v10 = 0;
    v11 = 0;
  }
  else
  {
    v8 = v17;
    v9 = v7 + 1;
    v10 = 0;
    if ( v9 <= v17 )
    {
      v8 = v9;
      v10 = v17 - v9;
    }
    v11 = v16 + v8;
  }
  v17 = v10;
  v16 = v11;
  LOBYTE(v18) = 45;
  v12 = sub_C931B0(&v16, &v18, 1u, 0);
  if ( v12 == -1 )
    return 0;
  v13 = v17;
  v14 = v12 + 1;
  if ( v14 <= v17 )
    v13 = v14;
  return v16 + v13;
}
