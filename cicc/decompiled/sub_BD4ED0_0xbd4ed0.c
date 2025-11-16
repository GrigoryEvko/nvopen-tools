// Function: sub_BD4ED0
// Address: 0xbd4ed0
//
__int64 __fastcall sub_BD4ED0(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r12

  v1 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 > 0x15u )
  {
    if ( v1 == 22 )
    {
      if ( (unsigned __int8)sub_B2BB70(a1) )
        return 0;
      v8 = *(_QWORD *)(a1 + 24);
      if ( (unsigned __int8)sub_B2DCE0(v8) || (unsigned __int8)sub_B2D610(v8, 29) )
      {
        if ( (unsigned __int8)sub_B2D610(v8, 39) )
          return 0;
      }
      if ( *(_BYTE *)a1 <= 0x1Cu )
      {
        if ( *(_BYTE *)a1 != 22 )
          return 1;
LABEL_7:
        v3 = *(_QWORD *)(a1 + 24);
LABEL_8:
        if ( !v3 )
          return 1;
        if ( (*(_BYTE *)(v3 + 3) & 0x40) == 0 )
          return 1;
        v4 = sub_B2DBE0(v3);
        if ( (unsigned int)sub_2241AC0(v4, "statepoint-example") || *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) >> 8 != 1 )
          return 1;
        v5 = *(_QWORD *)(v3 + 40);
        v6 = *(_QWORD *)(v5 + 32);
        v7 = v5 + 24;
        if ( v7 != v6 )
        {
          while ( 1 )
          {
            if ( !v6 )
              BUG();
            if ( *(_DWORD *)(v6 - 20) == 151 )
              break;
            v6 = *(_QWORD *)(v6 + 8);
            if ( v7 == v6 )
              return 0;
          }
          return 1;
        }
        return 0;
      }
    }
    else if ( v1 <= 0x1Cu )
    {
      return 1;
    }
    v3 = sub_B43CB0(a1);
    if ( *(_BYTE *)a1 != 22 )
      goto LABEL_8;
    goto LABEL_7;
  }
  return 0;
}
