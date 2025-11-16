// Function: sub_F8ECA0
// Address: 0xf8eca0
//
__int64 __fastcall sub_F8ECA0(unsigned __int8 *a1, __int64 a2)
{
  unsigned __int8 v4; // cl
  bool v5; // al
  unsigned __int8 *v6; // rdi

  if ( (unsigned __int8)sub_AC2D00((__int64)a1) || (unsigned __int8)sub_AC2D10((__int64)a1) )
    return 0;
  v4 = *a1;
  if ( *a1 == 18 )
    return sub_DFA980(a2);
  v5 = 1;
  if ( v4 <= 0x14u )
    v5 = ((0x120020uLL >> v4) & 1) == 0;
  if ( v4 > 3u && (unsigned __int8)(v4 - 12) > 1u && v5 )
    return 0;
  if ( v4 == 5 && ((v6 = sub_BD4070(a1, a2), v6 == a1) || !(unsigned __int8)sub_F8ECA0(v6, a2)) )
    return 0;
  else
    return sub_DFA980(a2);
}
