// Function: sub_14A8140
// Address: 0x14a8140
//
__int64 __fastcall sub_14A8140(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v4[2]; // [rsp-10h] [rbp-10h] BYREF

  if ( a1 == a2 )
    return a1;
  if ( !a1 || !a2 )
    return 0;
  v4[1] = v2;
  sub_14A8040(a1, a2, v4);
  return v4[0];
}
