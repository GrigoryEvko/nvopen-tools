// Function: sub_13CBE40
// Address: 0x13cbe40
//
__int64 __fastcall sub_13CBE40(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 result; // rax
  unsigned int v5; // r13d
  unsigned int v6; // r14d
  char v7; // r8

  if ( !a1 )
    return 0;
  v2 = *(_QWORD *)(a1 - 48);
  if ( !v2 )
    return 0;
  v3 = *(_QWORD *)(a1 - 24);
  if ( !v3 || !a2 || v2 != *(_QWORD *)(a2 - 48) || v3 != *(_QWORD *)(a2 - 24) )
    return 0;
  v5 = *(_WORD *)(a1 + 18) & 0x7FFF;
  v6 = *(_WORD *)(a2 + 18) & 0x7FFF;
  v7 = sub_15FF880(v5, v6);
  result = a2;
  if ( !v7 )
  {
    if ( (unsigned int)sub_15FF0F0(v6) != v5 )
    {
      if ( v5 != 33 )
      {
        if ( v6 == 39 && v5 == 41 || v5 == 37 && v6 == 35 )
          return sub_15A0600(*(_QWORD *)a1);
        return 0;
      }
      if ( !(unsigned __int8)sub_15FF820(v6) )
        return 0;
    }
    return sub_15A0600(*(_QWORD *)a1);
  }
  return result;
}
