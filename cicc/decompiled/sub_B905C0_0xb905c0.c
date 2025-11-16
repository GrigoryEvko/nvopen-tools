// Function: sub_B905C0
// Address: 0xb905c0
//
_BYTE *__fastcall sub_B905C0(_BYTE *a1)
{
  unsigned __int8 v1; // al
  __int64 v2; // rdi

  if ( *a1 == 16 )
    return a1;
  v1 = *(a1 - 16);
  if ( (v1 & 2) != 0 )
    v2 = *((_QWORD *)a1 - 4);
  else
    v2 = (__int64)&a1[-8 * ((v1 >> 2) & 0xF) - 16];
  return *(_BYTE **)v2;
}
