// Function: sub_28ED370
// Address: 0x28ed370
//
unsigned __int8 *__fastcall sub_28ED370(unsigned __int8 *a1, int a2, int a3)
{
  int v3; // eax
  unsigned __int8 *v4; // r12
  __int64 v6; // rcx
  int v7; // eax

  v3 = *a1;
  if ( (unsigned __int8)v3 <= 0x1Cu )
    return 0;
  if ( (unsigned int)(v3 - 42) > 0x11 )
    return 0;
  v6 = *((_QWORD *)a1 + 2);
  if ( !v6 )
    return 0;
  if ( *(_QWORD *)(v6 + 8) )
    return 0;
  v7 = v3 - 29;
  if ( a3 != v7 && a2 != v7 )
    return 0;
  v4 = a1;
  if ( (unsigned __int8)sub_920620((__int64)a1) )
  {
    if ( !sub_B451B0((__int64)a1) || !sub_B451E0((__int64)a1) )
      return 0;
  }
  return v4;
}
