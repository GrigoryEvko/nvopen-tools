// Function: sub_16E2230
// Address: 0x16e2230
//
__int64 __fastcall sub_16E2230(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v9; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v10; // [rsp+8h] [rbp-28h]
  __int64 v11; // [rsp+10h] [rbp-20h] BYREF
  unsigned __int64 v12; // [rsp+18h] [rbp-18h]

  v1 = *a1;
  LOBYTE(v9) = 45;
  v11 = v1;
  v12 = a1[1];
  v2 = sub_16D20C0(&v11, (char *)&v9, 1u, 0);
  if ( v2 == -1 )
  {
    v4 = 0;
    v5 = 0;
  }
  else
  {
    v3 = v2 + 1;
    if ( v3 > v12 )
      v3 = v12;
    v4 = v12 - v3;
    v5 = v11 + v3;
  }
  v10 = v4;
  v9 = v5;
  LOBYTE(v11) = 45;
  v6 = sub_16D20C0(&v9, (char *)&v11, 1u, 0);
  if ( v6 == -1 )
    return 0;
  v7 = v6 + 1;
  if ( v7 > v10 )
    v7 = v10;
  return v9 + v7;
}
