// Function: sub_CC75D0
// Address: 0xcc75d0
//
__int64 __fastcall sub_CC75D0(__int64 *a1)
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
  __int64 v11; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v12; // [rsp+8h] [rbp-28h]
  __int64 v13; // [rsp+10h] [rbp-20h] BYREF
  unsigned __int64 v14; // [rsp+18h] [rbp-18h]

  v1 = *a1;
  LOBYTE(v11) = 45;
  v13 = v1;
  v14 = a1[1];
  v2 = sub_C931B0(&v13, &v11, 1u, 0);
  if ( v2 == -1 )
  {
    v5 = 0;
    v6 = 0;
  }
  else
  {
    v3 = v14;
    v4 = v2 + 1;
    v5 = 0;
    if ( v4 <= v14 )
    {
      v3 = v4;
      v5 = v14 - v4;
    }
    v6 = v13 + v3;
  }
  v12 = v5;
  v11 = v6;
  LOBYTE(v13) = 45;
  v7 = sub_C931B0(&v11, &v13, 1u, 0);
  if ( v7 == -1 )
    return 0;
  v8 = v12;
  v9 = v7 + 1;
  if ( v9 <= v12 )
    v8 = v9;
  return v11 + v8;
}
