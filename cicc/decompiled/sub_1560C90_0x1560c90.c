// Function: sub_1560C90
// Address: 0x1560c90
//
_QWORD *__fastcall sub_1560C90(_QWORD *a1, __int64 a2, unsigned int *a3)
{
  __int64 v3; // rsi
  __int64 v4; // rax

  v3 = a2 << 32;
  v4 = 0xFFFFFFFFLL;
  if ( *((_BYTE *)a3 + 4) )
    v4 = *a3;
  return sub_1560C80(a1, v4 | v3);
}
