// Function: sub_12549C0
// Address: 0x12549c0
//
unsigned __int64 *__fastcall sub_12549C0(unsigned __int64 *a1, __int64 a2, _QWORD *a3, unsigned int a4)
{
  unsigned __int64 v6; // [rsp+8h] [rbp-28h] BYREF
  __int64 v7; // [rsp+10h] [rbp-20h] BYREF
  __int64 v8; // [rsp+18h] [rbp-18h]

  v7 = 0;
  v8 = 0;
  sub_1254950(&v6, a2, (__int64)&v7, a4);
  if ( (v6 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v6 & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  else
  {
    *a3 = v7;
    a3[1] = v8;
    *a1 = 1;
    return a1;
  }
}
