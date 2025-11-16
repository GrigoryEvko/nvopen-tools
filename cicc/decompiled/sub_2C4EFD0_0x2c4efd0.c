// Function: sub_2C4EFD0
// Address: 0x2c4efd0
//
__int64 __fastcall sub_2C4EFD0(_QWORD **a1, unsigned __int8 *a2)
{
  int v2; // eax
  int v4; // eax
  unsigned __int8 *v5; // rsi

  v2 = *a2;
  if ( (unsigned __int8)v2 > 0x1Cu )
  {
    v4 = v2 - 29;
  }
  else
  {
    if ( (_BYTE)v2 != 5 )
      return 0;
    v4 = *((unsigned __int16 *)a2 + 1);
  }
  if ( v4 != 49 )
    return 0;
  v5 = (a2[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a2 - 1) : &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  if ( !*(_QWORD *)v5 )
    return 0;
  **a1 = *(_QWORD *)v5;
  return 1;
}
