// Function: sub_1DD6020
// Address: 0x1dd6020
//
unsigned __int64 __fastcall sub_1DD6020(__int64 a1)
{
  __int64 v1; // r13
  unsigned __int64 v2; // r12
  __int64 v3; // r14
  unsigned __int64 v4; // rbx
  __int16 v5; // ax
  __int64 v6; // rax
  __int16 v7; // ax

  v1 = a1 + 24;
  v2 = a1 + 24;
  v3 = *(_QWORD *)(a1 + 32);
  if ( v3 != a1 + 24 )
  {
    while ( 1 )
    {
      v4 = *(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL;
      v5 = *(_WORD *)(v4 + 46);
      v2 = v4;
      if ( (v5 & 4) != 0 || (v5 & 8) == 0 )
        v6 = (*(_QWORD *)(*(_QWORD *)(v4 + 16) + 8LL) >> 6) & 1LL;
      else
        LOBYTE(v6) = sub_1E15D00(v4, 64, 1);
      if ( !(_BYTE)v6 && (unsigned __int16)(**(_WORD **)(v4 + 16) - 12) > 1u )
        break;
      if ( v3 == v4 )
        goto LABEL_12;
    }
    if ( v1 != v4 )
    {
      do
      {
LABEL_12:
        v7 = *(_WORD *)(v2 + 46);
        if ( (v7 & 4) != 0 || (v7 & 8) == 0 )
        {
          if ( (*(_QWORD *)(*(_QWORD *)(v2 + 16) + 8LL) & 0x40LL) != 0 )
            return v2;
        }
        else if ( (unsigned __int8)sub_1E15D00(v2, 64, 1) )
        {
          return v2;
        }
        v2 = *(_QWORD *)(v2 + 8);
      }
      while ( v1 != v2 );
    }
  }
  return v2;
}
