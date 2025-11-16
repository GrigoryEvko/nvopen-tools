// Function: sub_16E2000
// Address: 0x16e2000
//
__int64 __fastcall sub_16E2000(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v11; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v12; // [rsp+8h] [rbp-28h]
  __int64 v13; // [rsp+10h] [rbp-20h] BYREF
  unsigned __int64 v14; // [rsp+18h] [rbp-18h]

  v1 = *a1;
  LOBYTE(v11) = 45;
  v13 = v1;
  v14 = a1[1];
  v2 = sub_16D20C0(&v13, (char *)&v11, 1u, 0);
  if ( v2 == -1 )
  {
    v4 = 0;
    v5 = 0;
  }
  else
  {
    v3 = v2 + 1;
    if ( v3 > v14 )
      v3 = v14;
    v4 = v14 - v3;
    v5 = v13 + v3;
  }
  v12 = v4;
  v11 = v5;
  LOBYTE(v13) = 45;
  v6 = sub_16D20C0(&v11, (char *)&v13, 1u, 0);
  if ( v6 == -1 )
  {
    v8 = 0;
    v9 = 0;
  }
  else
  {
    v7 = v6 + 1;
    if ( v7 > v12 )
      v7 = v12;
    v8 = v12 - v7;
    v9 = v11 + v7;
  }
  v12 = v8;
  v11 = v9;
  LOBYTE(v13) = 45;
  if ( sub_16D20C0(&v11, (char *)&v13, 1u, 0) == -1 )
    return v11;
  else
    return v11;
}
