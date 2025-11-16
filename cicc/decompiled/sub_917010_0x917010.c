// Function: sub_917010
// Address: 0x917010
//
__int64 __fastcall sub_917010(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  char *v5; // rsi
  unsigned __int16 v6; // ax

  v3 = a3;
  if ( !a3 )
    v3 = sub_91A390(a1 + 8, *(_QWORD *)(a2 + 152), 0);
  v5 = (char *)sub_91B6C0(a2);
  if ( !*(_BYTE *)(a2 + 174) )
  {
    v6 = *(_WORD *)(a2 + 176);
    if ( v6 )
    {
      if ( v6 == 4741 )
      {
        v5 = "__ffsll";
      }
      else if ( v6 > 0x1285u )
      {
        switch ( v6 )
        {
          case 0x3CF2u:
            v5 = "__popcll";
            break;
          case 0x3D03u:
            v5 = "__ppc_trap";
            break;
          case 0x3CEEu:
            v5 = "__popc";
            break;
        }
      }
      else
      {
        switch ( v6 )
        {
          case 0x1167u:
            v5 = "__clzll";
            break;
          case 0x1281u:
            v5 = "__ffs";
            break;
          case 0x1163u:
            v5 = "__clz";
            break;
        }
      }
    }
  }
  return sub_9167F0(a1, v5, v3, a2);
}
