// Function: sub_3250780
// Address: 0x3250780
//
unsigned __int8 *__fastcall sub_3250780(__int64 *a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // rdx

  if ( !a2 )
    return (unsigned __int8 *)(a1 + 1);
  v2 = *(unsigned __int8 *)a2;
  if ( (unsigned __int8)(v2 - 16) <= 1u )
    return (unsigned __int8 *)(a1 + 1);
  if ( (unsigned __int8)v2 <= 0x24u )
  {
    v4 = 0x140000F000LL;
    if ( _bittest64(&v4, v2) )
      return (unsigned __int8 *)(*(__int64 (__fastcall **)(__int64 *))(*a1 + 40))(a1);
  }
  switch ( (_BYTE)v2 )
  {
    case 0x15:
      return (unsigned __int8 *)sub_324DC40(a1, a2);
    case 0x12:
      return sub_3250680(a1, (unsigned __int8 *)a2, 0);
    case 0x16:
      return (unsigned __int8 *)sub_324DE20(a1, a2);
  }
  return sub_3247C80((__int64)a1, (unsigned __int8 *)a2);
}
