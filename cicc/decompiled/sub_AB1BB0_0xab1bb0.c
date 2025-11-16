// Function: sub_AB1BB0
// Address: 0xab1bb0
//
__int64 __fastcall sub_AB1BB0(__int64 a1, __int64 a2)
{
  unsigned int v4; // r13d
  unsigned int v6; // eax
  bool v7; // al
  __int64 v8; // rdi
  __int64 v9; // rsi

  if ( sub_AAF760(a1) )
    return 1;
  LOBYTE(v6) = sub_AAF7D0(a2);
  v4 = v6;
  if ( (_BYTE)v6 )
    return 1;
  if ( sub_AAF7D0(a1) || sub_AAF760(a2) )
    return v4;
  if ( sub_AB0100(a1) )
  {
    v7 = sub_AB0100(a2);
    v8 = a2 + 16;
    v9 = a1 + 16;
    if ( v7 )
    {
      if ( (int)sub_C49970(v8, v9) > 0 )
        return v4;
      goto LABEL_13;
    }
    if ( (int)sub_C49970(v8, v9) > 0 )
    {
LABEL_13:
      LOBYTE(v4) = (int)sub_C49970(a1, a2) <= 0;
      return v4;
    }
    return 1;
  }
  if ( !sub_AB0100(a2) && (int)sub_C49970(a1, a2) <= 0 )
    LOBYTE(v4) = (int)sub_C49970(a2 + 16, a1 + 16) <= 0;
  return v4;
}
