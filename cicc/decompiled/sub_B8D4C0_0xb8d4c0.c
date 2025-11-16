// Function: sub_B8D4C0
// Address: 0xb8d4c0
//
bool __fastcall sub_B8D4C0(_BYTE *a1)
{
  unsigned __int8 v1; // al
  _BYTE **v3; // rdi

  if ( *a1 != 5 )
    return 0;
  v1 = *(a1 - 16);
  if ( (v1 & 2) != 0 )
  {
    if ( *((_DWORD *)a1 - 6) != 2 )
      return 0;
    v3 = (_BYTE **)*((_QWORD *)a1 - 4);
    if ( **v3 )
      return 0;
  }
  else
  {
    if ( ((*((_WORD *)a1 - 8) >> 6) & 0xF) != 2 )
      return 0;
    v3 = (_BYTE **)&a1[-8 * ((v1 >> 2) & 0xF) - 16];
    if ( **v3 )
      return 0;
  }
  return *v3[1] == 0;
}
