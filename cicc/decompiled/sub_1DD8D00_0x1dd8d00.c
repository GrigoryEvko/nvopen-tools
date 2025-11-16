// Function: sub_1DD8D00
// Address: 0x1dd8d00
//
char *__fastcall sub_1DD8D00(__int64 a1, char *a2)
{
  _BYTE *v2; // rsi
  char *result; // rax
  char *v4; // [rsp+8h] [rbp-8h] BYREF

  v4 = a2;
  v2 = *(_BYTE **)(a1 + 72);
  if ( v2 == *(_BYTE **)(a1 + 80) )
    return sub_1D4AF10(a1 + 64, v2, &v4);
  if ( v2 )
  {
    result = v4;
    *(_QWORD *)v2 = v4;
    v2 = *(_BYTE **)(a1 + 72);
  }
  *(_QWORD *)(a1 + 72) = v2 + 8;
  return result;
}
