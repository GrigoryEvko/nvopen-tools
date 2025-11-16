// Function: sub_16E1F20
// Address: 0x16e1f20
//
__int64 __fastcall sub_16E1F20(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v7[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v8; // [rsp+10h] [rbp-20h] BYREF
  unsigned __int64 v9; // [rsp+18h] [rbp-18h]

  v1 = *a1;
  LOBYTE(v7[0]) = 45;
  v8 = v1;
  v9 = a1[1];
  v2 = sub_16D20C0(&v8, (char *)v7, 1u, 0);
  if ( v2 == -1 )
  {
    v4 = 0;
    v5 = 0;
  }
  else
  {
    v3 = v2 + 1;
    if ( v3 > v9 )
      v3 = v9;
    v4 = v9 - v3;
    v5 = v8 + v3;
  }
  v7[1] = v4;
  v7[0] = v5;
  LOBYTE(v8) = 45;
  if ( sub_16D20C0(v7, (char *)&v8, 1u, 0) == -1 )
    return v7[0];
  else
    return v7[0];
}
