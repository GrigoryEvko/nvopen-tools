// Function: sub_F28360
// Address: 0xf28360
//
unsigned __int8 *__fastcall sub_F28360(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  char *v6; // rdx
  char v7; // al

  if ( **((_BYTE **)a2 - 4) > 0x15u )
    return 0;
  v6 = (char *)*((_QWORD *)a2 - 8);
  v7 = *v6;
  if ( (unsigned __int8)*v6 <= 0x1Cu )
    return 0;
  if ( v7 == 86 )
    return sub_F26350(a1, a2, (__int64)v6, 0);
  if ( v7 != 84 )
    return 0;
  return sub_F27020(a1, (__int64)a2, (__int64)v6, 0, a5, a6);
}
