// Function: sub_11FE380
// Address: 0x11fe380
//
unsigned __int64 __fastcall sub_11FE380(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3)
{
  __int64 v3; // rbp
  unsigned __int64 i; // r8
  unsigned __int64 v6; // rsi
  const char *v7; // [rsp-38h] [rbp-38h] BYREF
  char v8; // [rsp-18h] [rbp-18h]
  char v9; // [rsp-17h] [rbp-17h]
  __int64 v10; // [rsp-8h] [rbp-8h]

  if ( a2 == a3 )
    return 0;
  for ( i = (unsigned int)(__int16)word_3F64060[*a2]; a3 != ++a2; i = 16 * i + (unsigned int)(__int16)word_3F64060[*a2] )
  {
    if ( 16 * i + (unsigned int)(__int16)word_3F64060[*a2] < i )
    {
      v10 = v3;
      v6 = *(_QWORD *)(a1 + 56);
      v7 = "constant bigger than 64 bits detected";
      v9 = 1;
      v8 = 3;
      sub_11FD800(a1, v6, (__int64)&v7, 2);
      return 0;
    }
  }
  return i;
}
