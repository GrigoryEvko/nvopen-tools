// Function: sub_5CD6A0
// Address: 0x5cd6a0
//
__int64 __fastcall sub_5CD6A0(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rax
  _QWORD v6[3]; // [rsp+8h] [rbp-18h] BYREF

  v6[0] = a2;
  v4 = sub_5C7B50(a1, (__int64)v6, a3);
  if ( !v4 )
    return v6[0];
  if ( a3 != 6 )
  {
    *(_BYTE *)(*(_QWORD *)(v4 + 168) + 20LL) |= 8u;
    return v6[0];
  }
  sub_5CCAE0(5u, a1);
  return v6[0];
}
