// Function: sub_3184A00
// Address: 0x3184a00
//
__int64 __fastcall sub_3184A00(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  unsigned __int8 *v5; // rsi
  unsigned __int8 *v7; // rsi
  unsigned __int8 *v8; // rdx

  v5 = *(unsigned __int8 **)(a2 - 64);
  if ( *a3 == 86 && *(_QWORD *)(a2 - 96) == *((_QWORD *)a3 - 12) )
  {
    if ( !(unsigned __int8)sub_31843D0(a1, v5, *((unsigned __int8 **)a3 - 8)) )
    {
      v8 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
      v7 = *(unsigned __int8 **)(a2 - 32);
      return sub_31843D0(a1, v7, v8);
    }
    return 1;
  }
  if ( (unsigned __int8)sub_31843D0(a1, v5, a3) )
    return 1;
  v7 = *(unsigned __int8 **)(a2 - 32);
  v8 = a3;
  return sub_31843D0(a1, v7, v8);
}
