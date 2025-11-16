// Function: sub_29C0D10
// Address: 0x29c0d10
//
_BYTE *__fastcall sub_29C0D10(__int64 a1, unsigned int a2)
{
  __int128 v4; // rdi

  *(_QWORD *)&v4 = a1 + 160;
  *((_QWORD *)&v4 + 1) = a1;
  return sub_C54750(v4, a2);
}
