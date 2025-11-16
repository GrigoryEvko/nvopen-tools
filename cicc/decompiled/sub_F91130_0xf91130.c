// Function: sub_F91130
// Address: 0xf91130
//
__int64 __fastcall sub_F91130(unsigned __int8 *a1)
{
  unsigned __int8 v1; // al
  char *v4; // rdx
  char *v5; // rdi
  char v6; // al
  _BYTE *v7; // rdi

  v1 = *a1;
  if ( *a1 <= 0x1Cu )
    return 0;
  if ( (unsigned int)v1 - 42 <= 0x11 )
    return 1;
  if ( v1 != 86 )
    return 0;
  if ( (a1[7] & 0x40) != 0 )
  {
    v4 = (char *)*((_QWORD *)a1 - 1);
    v5 = (char *)*((_QWORD *)v4 + 4);
    v6 = *v5;
    if ( (unsigned __int8)*v5 > 0x15u )
      goto LABEL_9;
  }
  else
  {
    v4 = (char *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    v5 = (char *)*((_QWORD *)v4 + 4);
    v6 = *v5;
    if ( (unsigned __int8)*v5 > 0x15u )
      goto LABEL_9;
  }
  if ( v6 != 5 )
  {
    if ( !(unsigned __int8)sub_AD6CA0((__int64)v5) )
      return 1;
    if ( (a1[7] & 0x40) != 0 )
      v4 = (char *)*((_QWORD *)a1 - 1);
    else
      v4 = (char *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  }
LABEL_9:
  v7 = (_BYTE *)*((_QWORD *)v4 + 8);
  if ( *v7 > 0x15u || *v7 == 5 )
    return 0;
  else
    return (unsigned int)sub_AD6CA0((__int64)v7) ^ 1;
}
