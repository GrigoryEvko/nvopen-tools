// Function: sub_370F8F0
// Address: 0x370f8f0
//
unsigned __int64 *__fastcall sub_370F8F0(
        unsigned __int64 *a1,
        _QWORD *a2,
        char *a3,
        const __m128i *a4,
        unsigned int a5)
{
  __int64 v7; // rax
  char v9; // [rsp+7h] [rbp-29h] BYREF
  unsigned __int64 v10[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( a2[7] && !a2[5] && !a2[6] )
  {
LABEL_13:
    if ( !a2[5] )
      v9 = *a3;
    goto LABEL_6;
  }
  if ( !(unsigned int)sub_3700ED0((__int64)a2, (__int64)a2, (__int64)a3, (__int64)a4, a5) )
  {
    sub_370CCD0(a1, 2u);
    return a1;
  }
  v7 = a2[7];
  if ( a2[6] )
  {
    if ( !v7 )
      goto LABEL_13;
  }
  else if ( v7 )
  {
    goto LABEL_13;
  }
LABEL_6:
  sub_3702900(v10, a2, &v9, a4);
  if ( (v10[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v10[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  else
  {
    if ( a2[5] && !a2[7] && !a2[6] )
      *a3 = v9;
    *a1 = 1;
    return a1;
  }
}
