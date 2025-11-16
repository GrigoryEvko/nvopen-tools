// Function: sub_16F23B0
// Address: 0x16f23b0
//
__int64 __fastcall sub_16F23B0(unsigned __int8 *a1, __int64 a2, _QWORD *a3, __int64 a4, int a5)
{
  unsigned __int8 *v5; // rbp
  unsigned __int8 *v6; // rsi
  char *v7; // rax
  __int64 result; // rax
  unsigned __int8 *v10[4]; // [rsp-20h] [rbp-20h] BYREF

  v6 = &a1[a2];
  if ( a1 == v6 )
    return 1;
  v10[3] = v5;
  v7 = (char *)a1;
  while ( *v7 >= 0 )
  {
    if ( v6 == (unsigned __int8 *)++v7 )
      return 1;
  }
  v10[0] = a1;
  if ( (unsigned __int8)sub_16F0F00(v10, v6, (__int64)a3, a4, a5) )
    return 1;
  result = 0;
  if ( a3 )
    *a3 = v10[0] - a1;
  return result;
}
