// Function: sub_24C09A0
// Address: 0x24c09a0
//
__int64 __fastcall sub_24C09A0(__int64 a1)
{
  __int64 i; // rbx
  unsigned __int8 *v2; // r12
  unsigned __int8 v3; // al
  __int64 v5; // rax

  for ( i = *(_QWORD *)(a1 + 16); i; i = *(_QWORD *)(i + 8) )
  {
    v2 = *(unsigned __int8 **)(i + 24);
    if ( *v2 <= 0x1Cu )
      return 1;
    if ( !sub_B46A10(*(_QWORD *)(i + 24)) && !sub_BD2BE0((__int64)v2) )
    {
      v3 = *v2;
      if ( *v2 == 85 )
      {
        if ( (unsigned __int8)sub_24C0860((__int64)v2) )
          continue;
        v3 = *v2;
      }
      if ( v3 <= 0x1Cu )
        return 1;
      if ( v3 != 61 )
      {
        if ( v3 == 62 )
        {
          v5 = *((_QWORD *)v2 - 4);
          if ( a1 != v5 || !v5 )
            return 1;
        }
        else if ( v3 != 63 && v3 != 78 || (unsigned __int8)sub_24C09A0(v2) )
        {
          return 1;
        }
      }
    }
  }
  return 0;
}
