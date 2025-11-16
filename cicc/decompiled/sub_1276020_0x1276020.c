// Function: sub_1276020
// Address: 0x1276020
//
__int64 __fastcall sub_1276020(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  char *v7; // rsi
  unsigned __int16 v8; // ax

  v5 = a3;
  if ( !a3 )
    v5 = sub_127A030(a1 + 8, *(_QWORD *)(a2 + 152), 0, a4, a5);
  v7 = (char *)sub_127B370(a2);
  if ( !*(_BYTE *)(a2 + 174) )
  {
    v8 = *(_WORD *)(a2 + 176);
    if ( v8 )
    {
      if ( v8 == 4741 )
      {
        v7 = "__ffsll";
      }
      else if ( v8 > 0x1285u )
      {
        switch ( v8 )
        {
          case 0x3CF2u:
            v7 = "__popcll";
            break;
          case 0x3D03u:
            v7 = "__ppc_trap";
            break;
          case 0x3CEEu:
            v7 = "__popc";
            break;
        }
      }
      else
      {
        switch ( v8 )
        {
          case 0x1167u:
            v7 = "__clzll";
            break;
          case 0x1281u:
            v7 = "__ffs";
            break;
          case 0x1163u:
            v7 = "__clz";
            break;
        }
      }
    }
  }
  return sub_1275780(a1, v7, v5, a2);
}
