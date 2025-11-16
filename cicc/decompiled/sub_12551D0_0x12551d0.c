// Function: sub_12551D0
// Address: 0x12551d0
//
unsigned __int64 *__fastcall sub_12551D0(unsigned __int64 *a1, __int64 a2, unsigned __int64 a3, _QWORD *a4)
{
  unsigned __int64 v6; // rax
  __int64 v8; // rdx
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = *(_QWORD *)(a2 + 48);
  if ( a3 > v6 )
  {
    sub_1254FA0(v9, 3);
  }
  else
  {
    if ( a3 + 1 <= v6 )
      goto LABEL_7;
    sub_1254FA0(v9, 1);
  }
  if ( (v9[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v9[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  v6 = *(_QWORD *)(a2 + 48);
LABEL_7:
  v8 = *(_QWORD *)(a2 + 40);
  a4[1] = v6 - a3;
  *a4 = a3 + v8;
  *a1 = 1;
  return a1;
}
