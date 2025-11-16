// Function: sub_AB1B10
// Address: 0xab1b10
//
char __fastcall sub_AB1B10(__int64 a1, __int64 a2)
{
  int v2; // eax
  bool v3; // r13

  if ( *(_DWORD *)(a1 + 8) <= 0x40u )
  {
    if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 16) )
      goto LABEL_3;
LABEL_5:
    v3 = sub_AB0100(a1);
    if ( v3 )
    {
      if ( (int)sub_C49970(a1, a2) <= 0 )
        goto LABEL_7;
    }
    else if ( (int)sub_C49970(a1, a2) > 0 )
    {
LABEL_7:
      LOBYTE(v2) = v3;
      return v2;
    }
    return (unsigned int)sub_C49970(a2, a1 + 16) >> 31;
  }
  if ( !(unsigned __int8)sub_C43C50(a1, a1 + 16) )
    goto LABEL_5;
LABEL_3:
  LOBYTE(v2) = sub_AAF760(a1);
  return v2;
}
