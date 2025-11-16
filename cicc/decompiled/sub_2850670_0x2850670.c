// Function: sub_2850670
// Address: 0x2850670
//
char __fastcall sub_2850670(
        __int64 *a1,
        __int64 a2,
        char a3,
        __int64 a4,
        char a5,
        unsigned int a6,
        __int64 a7,
        __int64 a8,
        unsigned __int64 a9,
        __int64 a10,
        char a11,
        unsigned __int8 a12,
        unsigned __int64 a13)
{
  char v18; // al
  __int64 v19; // rcx

  if ( a10 && (a11 != a3 || a3 != a5) )
    return 0;
  if ( a2 + a10 > a10 != a2 > 0 )
    return 0;
  if ( a4 + a10 > a10 != a4 > 0 )
    return 0;
  v18 = sub_2850560(a1, a6, a7, (unsigned int)a8, a9, a2 + a10, a3, a12, a13);
  v19 = (unsigned int)a8;
  if ( !v18 )
    return 0;
  LODWORD(a8) = a12;
  return sub_2850560(a1, a6, a7, v19, a9, a4 + a10, a5, a8, a13);
}
