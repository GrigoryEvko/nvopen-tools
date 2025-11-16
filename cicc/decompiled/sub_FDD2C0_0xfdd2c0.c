// Function: sub_FDD2C0
// Address: 0xfdd2c0
//
__int64 __fastcall sub_FDD2C0(__int64 *a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v3; // r13
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v8; // [rsp+10h] [rbp-40h] BYREF
  __int64 v9; // [rsp+18h] [rbp-38h]

  v3 = *a1;
  if ( *a1 )
  {
    v5 = sub_FDC440(a1);
    LODWORD(v8) = sub_FDD0F0(v3, a2);
    v8 = sub_FE8990(v3, v5, &v8, a3);
    v9 = v6;
  }
  else
  {
    LOBYTE(v9) = 0;
  }
  return v8;
}
