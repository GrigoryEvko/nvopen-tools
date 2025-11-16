// Function: sub_12BD800
// Address: 0x12bd800
//
__int64 __fastcall sub_12BD800(__int64 a1, const char *a2)
{
  size_t v2; // rdx
  void *v4; // [rsp+0h] [rbp-40h] BYREF
  __int64 v5; // [rsp+8h] [rbp-38h]
  __int64 v6; // [rsp+10h] [rbp-30h]
  __int64 v7; // [rsp+18h] [rbp-28h]
  int v8; // [rsp+20h] [rbp-20h]
  __int64 v9; // [rsp+28h] [rbp-18h]

  v9 = a1;
  v8 = 1;
  v7 = 0;
  v6 = 0;
  v5 = 0;
  v4 = &unk_49EFBE0;
  if ( a2 )
  {
    v2 = strlen(a2);
    if ( v2 )
    {
      sub_16E7EE0(&v4, a2, v2);
      if ( v7 != v5 )
        sub_16E7BA0(&v4);
    }
  }
  return sub_16E7BC0(&v4);
}
