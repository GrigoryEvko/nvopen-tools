// Function: sub_29E18A0
// Address: 0x29e18a0
//
__int64 __fastcall sub_29E18A0(__int64 a1, _BYTE *a2)
{
  __int64 v2; // r12
  __int64 *v3; // rbx
  __int64 v4; // r15
  __int64 v6; // [rsp+8h] [rbp-48h]
  __int64 v7; // [rsp+10h] [rbp-40h] BYREF
  __int64 v8[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = (__int64)a2;
  if ( a2 && *a2 == 6 )
  {
    v3 = *(__int64 **)a1;
    v4 = **(_QWORD **)(a1 + 8);
    v6 = *(_QWORD *)(a1 + 16);
    sub_B10CB0(&v7, (__int64)a2);
    sub_29E11F0(v8, (__int64)&v7, v4, v3, v6);
    v2 = sub_B10CD0((__int64)v8);
    if ( v8[0] )
      sub_B91220((__int64)v8, v8[0]);
    if ( v7 )
      sub_B91220((__int64)&v7, v7);
  }
  return v2;
}
