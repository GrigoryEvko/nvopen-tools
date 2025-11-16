// Function: sub_15B0C30
// Address: 0x15b0c30
//
const char *__fastcall sub_15B0C30(unsigned __int8 *a1)
{
  unsigned __int8 v1; // al
  __int64 v3; // rdi

  v1 = *a1;
  if ( *a1 <= 0xEu )
  {
    if ( v1 <= 0xAu )
      return byte_3F871B3;
LABEL_6:
    v3 = *(_QWORD *)&a1[8 * (2LL - *((unsigned int *)a1 + 2))];
    if ( !v3 )
      return (const char *)v3;
    return (const char *)sub_161E970(v3);
  }
  if ( (unsigned __int8)(v1 - 32) <= 1u || v1 == 17 || v1 == 20 || v1 == 31 )
    goto LABEL_6;
  if ( v1 != 21 )
    return byte_3F871B3;
  v3 = *(_QWORD *)&a1[8 * (1LL - *((unsigned int *)a1 + 2))];
  if ( v3 )
    return (const char *)sub_161E970(v3);
  return (const char *)v3;
}
