// Function: sub_DC3660
// Address: 0xdc3660
//
__int64 __fastcall sub_DC3660(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rax
  __int64 v5; // r12
  __int64 *v6; // rax
  __int64 v8; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v9; // [rsp+18h] [rbp-98h]
  __int64 v10; // [rsp+20h] [rbp-90h]
  __int64 v11; // [rsp+28h] [rbp-88h] BYREF
  unsigned int v12; // [rsp+30h] [rbp-80h]
  __int64 v13; // [rsp+68h] [rbp-48h] BYREF
  __int16 v14; // [rsp+70h] [rbp-40h]

  v3 = &v11;
  v8 = a1;
  v9 = 0;
  v10 = 1;
  do
  {
    *v3 = -4096;
    v3 += 2;
  }
  while ( v3 != &v13 );
  v14 = 0;
  v13 = a2;
  v5 = sub_DD45D0(&v8, a3);
  if ( (_BYTE)v14 )
    v5 = sub_D970F0(a1);
  if ( (v10 & 1) == 0 )
    sub_C7D6A0(v11, 16LL * v12, 8);
  if ( v5 != sub_D970F0(a1) )
  {
    v8 = a1;
    v6 = &v11;
    v9 = 0;
    v10 = 1;
    do
    {
      *v6 = -4096;
      v6 += 2;
    }
    while ( v6 != &v13 );
    v13 = a2;
    v14 = 0;
    sub_DC2D20(&v8, a3);
    if ( (_BYTE)v14 )
    {
      sub_D970F0(a1);
      if ( (v10 & 1) != 0 )
        return v5;
    }
    else if ( (v10 & 1) != 0 )
    {
      return v5;
    }
    sub_C7D6A0(v11, 16LL * v12, 8);
  }
  return v5;
}
