// Function: sub_14A9510
// Address: 0x14a9510
//
unsigned __int64 *__fastcall sub_14A9510(unsigned __int64 *a1, __int64 a2, unsigned __int64 *a3, unsigned int a4)
{
  __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned __int64 v8; // rdx

  v6 = *((unsigned int *)a3 + 2);
  *((_DWORD *)a1 + 2) = v6;
  if ( (unsigned int)v6 > 0x40 )
  {
    sub_16A4FD0(a1, a3);
    v6 = *((unsigned int *)a1 + 2);
    if ( (unsigned int)v6 > 0x40 )
    {
      sub_16A8110(a1, a4);
      v6 = *((unsigned int *)a1 + 2);
      goto LABEL_5;
    }
  }
  else
  {
    *a1 = *a3;
  }
  if ( a4 == (_DWORD)v6 )
  {
    *a1 = 0;
    v8 = 0;
    LOBYTE(v7) = 0;
    if ( !a4 )
      return a1;
    goto LABEL_9;
  }
  *a1 >>= a4;
LABEL_5:
  v7 = (unsigned int)v6 - a4;
  if ( (_DWORD)v7 == (_DWORD)v6 )
    return a1;
  if ( (unsigned int)v7 <= 0x3F && (unsigned int)v6 <= 0x40 )
  {
    v8 = *a1;
LABEL_9:
    *a1 = v8 | (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)a4) << v7);
    return a1;
  }
  sub_16A5260(a1, v7, v6);
  return a1;
}
