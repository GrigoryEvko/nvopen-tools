// Function: sub_317EC90
// Address: 0x317ec90
//
__int64 __fastcall sub_317EC90(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // r12

  v2 = sub_317E6A0(a1, a2);
  if ( !v2 )
    return 0;
  v3 = v2;
  v4 = sub_317E470(v2);
  if ( !v4 )
    return 0;
  if ( a1 + 120 != sub_317E650(v3) )
  {
    *(_DWORD *)(v4 + 48) |= 4u;
    return v4;
  }
  return v4;
}
