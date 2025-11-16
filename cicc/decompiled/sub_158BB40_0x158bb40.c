// Function: sub_158BB40
// Address: 0x158bb40
//
__int64 __fastcall sub_158BB40(__int64 a1, __int64 a2)
{
  unsigned int v4; // r13d
  unsigned int v6; // eax
  bool v7; // al
  __int64 v8; // rsi
  __int64 v9; // rdi

  if ( sub_158A0B0(a1) )
    return 1;
  LOBYTE(v6) = sub_158A120(a2);
  v4 = v6;
  if ( (_BYTE)v6 )
    return 1;
  if ( sub_158A120(a1) || sub_158A0B0(a2) )
    return v4;
  if ( sub_158A670(a1) )
  {
    v7 = sub_158A670(a2);
    v8 = a1 + 16;
    v9 = a2 + 16;
    if ( v7 )
    {
      if ( (int)sub_16A9900(v9, v8) > 0 )
        return v4;
      goto LABEL_13;
    }
    if ( (int)sub_16A9900(v9, v8) > 0 )
    {
LABEL_13:
      LOBYTE(v4) = (int)sub_16A9900(a1, a2) <= 0;
      return v4;
    }
    return 1;
  }
  if ( !sub_158A670(a2) && (int)sub_16A9900(a1, a2) <= 0 )
    LOBYTE(v4) = (int)sub_16A9900(a2 + 16, a1 + 16) <= 0;
  return v4;
}
