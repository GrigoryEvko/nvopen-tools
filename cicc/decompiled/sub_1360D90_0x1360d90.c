// Function: sub_1360D90
// Address: 0x1360d90
//
__int64 __fastcall sub_1360D90(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 *v5; // r10
  __int64 v6; // r8
  bool v7; // r9
  __int64 *v8; // r10
  __int64 result; // rax
  char v10; // r8

  if ( sub_135D850(a2, 4) || sub_135D850(v2, 4) )
    return 4;
  if ( !sub_135D850(v3, 79) )
  {
    v7 = sub_135D850(v4, 79);
    result = 7;
    if ( !v7 )
      return result;
    if ( (sub_1360180(v8, v6) & 2) != 0 )
      return 6;
    return 4;
  }
  v10 = sub_1360180(v5, v4);
  result = 5;
  if ( (v10 & 2) == 0 )
    return 4;
  return result;
}
