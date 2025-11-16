// Function: sub_28ED300
// Address: 0x28ed300
//
unsigned __int8 *__fastcall sub_28ED300(unsigned __int8 *a1, int a2)
{
  int v2; // eax
  unsigned __int8 *v3; // r12
  __int64 v5; // rdx

  v2 = *a1;
  if ( (unsigned __int8)v2 <= 0x1Cu )
    return 0;
  if ( (unsigned int)(v2 - 42) > 0x11 )
    return 0;
  v5 = *((_QWORD *)a1 + 2);
  if ( !v5 )
    return 0;
  if ( *(_QWORD *)(v5 + 8) )
    return 0;
  if ( a2 != v2 - 29 )
    return 0;
  v3 = a1;
  if ( (unsigned __int8)sub_920620((__int64)a1) )
  {
    if ( !sub_B451B0((__int64)a1) || !sub_B451E0((__int64)a1) )
      return 0;
  }
  return v3;
}
