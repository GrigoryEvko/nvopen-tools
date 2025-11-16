// Function: sub_2853CD0
// Address: 0x2853cd0
//
__int64 __fastcall sub_2853CD0(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // r12
  __int64 **v4; // rax
  unsigned __int64 v5; // [rsp+0h] [rbp-30h] BYREF
  __int64 v6; // [rsp+8h] [rbp-28h]
  __int64 v7; // [rsp+10h] [rbp-20h]

  v2 = a1[2];
  if ( v2 )
  {
    v5 = 4;
    v6 = 0;
    v7 = v2;
    if ( v2 == -4096 || v2 == -8192 )
      return v2;
    sub_BD6050(&v5, *a1 & 0xFFFFFFFFFFFFFFF8LL);
  }
  else
  {
    v4 = (__int64 **)sub_BCB2B0(a2);
    v5 = 4;
    v7 = sub_ACADE0(v4);
    v2 = v7;
    v6 = 0;
    if ( v7 == 0 || v7 == -4096 || v7 == -8192 )
      return v2;
    sub_BD73F0((__int64)&v5);
  }
  v2 = v7;
  if ( v7 != 0 && v7 != -4096 && v7 != -8192 )
    sub_BD60C0(&v5);
  return v2;
}
