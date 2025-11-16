// Function: sub_9C8BE0
// Address: 0x9c8be0
//
__int64 *__fastcall sub_9C8BE0(__int64 *a1, __int64 a2, __int64 a3, char a4)
{
  const char *v4; // rax
  const char *v6; // [rsp+0h] [rbp-40h] BYREF
  char v7; // [rsp+20h] [rbp-20h]
  char v8; // [rsp+21h] [rbp-1Fh]

  if ( a4 != 14 )
  {
    v8 = 1;
    v4 = "Load/Store operand is not a pointer type";
LABEL_3:
    v6 = v4;
    v7 = 3;
    sub_9C81F0(a1, a2 + 8, (__int64)&v6);
    return a1;
  }
  if ( !(unsigned __int8)sub_BCBD30(a3) )
  {
    v8 = 1;
    v4 = "Cannot load/store from pointer";
    goto LABEL_3;
  }
  *a1 = 1;
  return a1;
}
