// Function: sub_1DD5EE0
// Address: 0x1dd5ee0
//
unsigned __int64 __fastcall sub_1DD5EE0(__int64 a1)
{
  __int64 v1; // rbx
  unsigned __int64 v2; // r12
  __int64 v3; // r13
  unsigned __int64 v4; // rdx
  __int16 v5; // ax
  __int64 v6; // rax
  __int16 v7; // ax

  v1 = a1 + 24;
  v2 = a1 + 24;
  v3 = *(_QWORD *)(a1 + 32);
  if ( v3 == a1 + 24 )
    return v2;
  do
  {
    v4 = *(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL;
    v5 = *(_WORD *)(v4 + 46);
    v2 = v4;
    if ( v4 && (*(_BYTE *)v4 & 4) != 0 )
    {
      if ( (v5 & 4) != 0 )
        goto LABEL_5;
    }
    else
    {
      while ( (v5 & 4) != 0 )
      {
        v2 = *(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL;
        v5 = *(_WORD *)(v2 + 46);
      }
    }
    if ( (v5 & 8) != 0 )
    {
      LOBYTE(v6) = sub_1E15D00(v2, 64, 1);
      goto LABEL_6;
    }
LABEL_5:
    v6 = (*(_QWORD *)(*(_QWORD *)(v2 + 16) + 8LL) >> 6) & 1LL;
LABEL_6:
    if ( !(_BYTE)v6 && (unsigned __int16)(**(_WORD **)(v2 + 16) - 12) > 1u )
    {
      if ( v1 != v2 )
        goto LABEL_14;
      return v2;
    }
  }
  while ( v3 != v2 );
  v7 = *(_WORD *)(v2 + 46);
  if ( (v7 & 4) != 0 )
    goto LABEL_15;
LABEL_10:
  if ( (v7 & 8) != 0 )
  {
    if ( !(unsigned __int8)sub_1E15D00(v2, 64, 1) )
      goto LABEL_12;
  }
  else
  {
LABEL_15:
    while ( (*(_QWORD *)(*(_QWORD *)(v2 + 16) + 8LL) & 0x40LL) == 0 )
    {
LABEL_12:
      if ( (*(_BYTE *)v2 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v2 + 46) & 8) != 0 )
          v2 = *(_QWORD *)(v2 + 8);
      }
      v2 = *(_QWORD *)(v2 + 8);
      if ( v2 == v1 )
        return v2;
LABEL_14:
      v7 = *(_WORD *)(v2 + 46);
      if ( (v7 & 4) == 0 )
        goto LABEL_10;
    }
  }
  return v2;
}
