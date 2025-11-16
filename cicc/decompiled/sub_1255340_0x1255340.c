// Function: sub_1255340
// Address: 0x1255340
//
unsigned __int64 *__fastcall sub_1255340(
        unsigned __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        const void *a4,
        size_t a5)
{
  unsigned __int64 v8; // rax
  __int64 v10[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( !a5 )
    goto LABEL_8;
  v8 = sub_1254EB0(a2 + 8);
  if ( a3 > v8 )
  {
    sub_1254FA0(v10, 3);
  }
  else
  {
    if ( v8 >= a3 + a5 )
    {
LABEL_7:
      memcpy((void *)(a3 + *(_QWORD *)(a2 + 16)), a4, a5);
LABEL_8:
      *a1 = 1;
      return a1;
    }
    sub_1254FA0(v10, 1);
  }
  if ( (v10[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    goto LABEL_7;
  *a1 = v10[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  return a1;
}
