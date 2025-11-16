// Function: sub_DD86E0
// Address: 0xdd86e0
//
__int64 __fastcall sub_DD86E0(__int64 *a1, _BYTE *a2)
{
  bool v2; // r13
  bool v3; // al
  unsigned __int8 *v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  unsigned int v7; // r13d

  if ( *a2 == 5 )
    return 0;
  v2 = sub_B448F0((__int64)a2);
  v3 = sub_B44900((__int64)a2);
  if ( v2 )
  {
    v7 = !v3 ? 2 : 6;
  }
  else
  {
    if ( !v3 )
      return 0;
    v7 = 4;
  }
  if ( !(unsigned __int8)sub_DD8590(a1, (__int64)a2, v4, v5, v6) )
    return 0;
  return v7;
}
