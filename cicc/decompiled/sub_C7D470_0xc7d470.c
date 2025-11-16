// Function: sub_C7D470
// Address: 0xc7d470
//
_QWORD *__fastcall sub_C7D470(_QWORD *a1, unsigned __int8 *a2)
{
  _QWORD *v2; // r9
  __int64 v3; // rax
  unsigned __int8 v4; // dl
  unsigned __int8 v5; // cl
  char v6; // dl

  v2 = a1 + 3;
  a1[2] = 32;
  v3 = 0;
  *a1 = a1 + 3;
  a1[1] = 32;
  while ( 1 )
  {
    v4 = *a2++;
    v5 = v4;
    v6 = a0123456789abcd_10[v4 & 0xF] | 0x20;
    *((_BYTE *)v2 + v3) = a0123456789abcd_10[v5 >> 4] | 0x20;
    *(_BYTE *)(*a1 + v3 + 1) = v6;
    v3 += 2;
    if ( v3 == 32 )
      break;
    v2 = (_QWORD *)*a1;
  }
  return a1;
}
