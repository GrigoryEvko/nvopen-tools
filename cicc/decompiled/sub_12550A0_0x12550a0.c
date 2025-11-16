// Function: sub_12550A0
// Address: 0x12550a0
//
unsigned __int64 *__fastcall sub_12550A0(
        unsigned __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 *a5)
{
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rbx
  __int64 v11[7]; // [rsp+8h] [rbp-38h] BYREF

  v8 = *(_QWORD *)(a2 + 48);
  if ( a3 > v8 )
  {
    sub_1254FA0(v11, 3);
  }
  else
  {
    if ( a4 + a3 <= v8 )
    {
LABEL_6:
      v9 = *(_QWORD *)(a2 + 40) + a3;
      a5[1] = a4;
      *a5 = v9;
      *a1 = 1;
      return a1;
    }
    sub_1254FA0(v11, 1);
  }
  if ( (v11[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    goto LABEL_6;
  *a1 = v11[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  return a1;
}
