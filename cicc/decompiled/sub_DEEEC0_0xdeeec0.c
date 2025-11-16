// Function: sub_DEEEC0
// Address: 0xdeeec0
//
__int64 __fastcall sub_DEEEC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rax
  __int64 v7; // r12
  _QWORD v9[2]; // [rsp+0h] [rbp-80h] BYREF
  __int64 v10; // [rsp+10h] [rbp-70h]
  __int64 v11; // [rsp+18h] [rbp-68h] BYREF
  unsigned int v12; // [rsp+20h] [rbp-60h]
  _QWORD v13[4]; // [rsp+58h] [rbp-28h] BYREF

  v6 = &v11;
  v9[0] = a1;
  v9[1] = 0;
  v10 = 1;
  do
  {
    *v6 = -4096;
    v6 += 2;
  }
  while ( v6 != v13 );
  v13[2] = a3;
  v13[0] = 0;
  v13[1] = a4;
  v7 = sub_DE9D10(v9, a2, (__int64)v13, a4, a1, a6);
  if ( (v10 & 1) == 0 )
    sub_C7D6A0(v11, 16LL * v12, 8);
  return v7;
}
