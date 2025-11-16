// Function: sub_C85CF0
// Address: 0xc85cf0
//
__int64 sub_C85CF0()
{
  __int64 result; // rax
  __int64 v1; // rax
  __int64 v2; // [rsp+10h] [rbp-40h]
  __int64 v3; // [rsp+18h] [rbp-38h]
  __int64 v4; // [rsp+20h] [rbp-30h] BYREF
  __int64 v5; // [rsp+28h] [rbp-28h] BYREF
  __int64 v6; // [rsp+30h] [rbp-20h] BYREF
  char v7; // [rsp+38h] [rbp-18h]

  sub_C85FC0(&v6);
  if ( (v7 & 1) == 0 )
    return (unsigned int)v6;
  v2 = 0;
  v7 &= ~2u;
  v1 = v6;
  v6 = 0;
  v3 = 0;
  v5 = v1 | 1;
  sub_C85B20(&v4, &v5);
  if ( (v4 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    BUG();
  if ( (v5 & 1) != 0 || (v5 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v5, (__int64)&v5);
  if ( (v7 & 2) != 0 )
    sub_9CE230(&v6);
  if ( (v7 & 1) == 0 )
    return 4096;
  result = 4096;
  if ( v6 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
    return 4096;
  }
  return result;
}
