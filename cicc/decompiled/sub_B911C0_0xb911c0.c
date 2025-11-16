// Function: sub_B911C0
// Address: 0xb911c0
//
unsigned __int8 *__fastcall sub_B911C0(unsigned __int8 *a1)
{
  int v1; // eax
  unsigned __int64 v2; // r8

  v1 = *a1;
  if ( (unsigned __int8)(v1 - 5) <= 0x1Fu )
  {
    if ( (a1[1] & 0x7F) == 2 || *((_DWORD *)a1 - 2) || (v2 = 0, (_BYTE)v1 == 30) )
    {
      v2 = *((_QWORD *)a1 + 1) & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*((_QWORD *)a1 + 1) & 4) == 0 )
        return 0;
    }
    return (unsigned __int8 *)v2;
  }
  if ( (_BYTE)v1 != 4 && (unsigned int)(v1 - 1) > 1 )
    return 0;
  return a1 + 8;
}
