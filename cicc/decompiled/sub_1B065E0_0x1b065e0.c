// Function: sub_1B065E0
// Address: 0x1b065e0
//
char *__fastcall sub_1B065E0(__int64 a1, char **a2)
{
  _BYTE *v3; // rsi
  char *result; // rax

  v3 = *(_BYTE **)(a1 + 8);
  if ( v3 == *(_BYTE **)(a1 + 16) )
    return sub_15D0700(a1, v3, a2);
  if ( v3 )
  {
    result = *a2;
    *(_QWORD *)v3 = *a2;
    v3 = *(_BYTE **)(a1 + 8);
  }
  *(_QWORD *)(a1 + 8) = v3 + 8;
  return result;
}
