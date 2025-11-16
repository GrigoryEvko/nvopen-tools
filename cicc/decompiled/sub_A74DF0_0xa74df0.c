// Function: sub_A74DF0
// Address: 0xa74df0
//
__int64 __fastcall sub_A74DF0(__int64 a1, int a2)
{
  __int64 result; // rax
  __int64 v3; // [rsp+18h] [rbp-18h] BYREF
  __int64 v4; // [rsp+20h] [rbp-10h]
  __int64 v5; // [rsp+28h] [rbp-8h]

  v3 = sub_A74D20(a1, a2);
  if ( !v3 )
    return v4;
  result = sub_A71B80(&v3);
  LOBYTE(v5) = 1;
  v4 = result;
  return result;
}
