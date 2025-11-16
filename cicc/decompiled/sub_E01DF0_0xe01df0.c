// Function: sub_E01DF0
// Address: 0xe01df0
//
__int64 __fastcall sub_E01DF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbp
  _QWORD v8[2]; // [rsp-10h] [rbp-10h] BYREF

  if ( a1 == a2 )
    return a1;
  if ( !a1 || !a2 )
    return 0;
  v8[1] = v6;
  sub_E018D0(a1, a2, (__int64)v8, a4, a5, a6);
  return v8[0];
}
