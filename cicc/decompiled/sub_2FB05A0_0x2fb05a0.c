// Function: sub_2FB05A0
// Address: 0x2fb05a0
//
__int64 __fastcall sub_2FB05A0(_QWORD *a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  char *v7; // rsi
  char *i; // rsi
  __int64 v10; // r13
  __int64 v11; // rsi

  result = 0x800000000LL;
  *a1 = a2;
  v7 = (char *)(a1 + 3);
  a1[1] = a1 + 3;
  a1[2] = 0x800000000LL;
  if ( a3 )
  {
    result = (__int64)(a1 + 3);
    if ( a3 > 8uLL )
    {
      v10 = 16LL * a3;
      sub_C8D5F0((__int64)(a1 + 1), v7, a3, 0x10u, a5, a6);
      v11 = a1[1];
      result = v11 + 16LL * *((unsigned int *)a1 + 4);
      for ( i = (char *)(v10 + v11); i != (char *)result; result += 16 )
      {
LABEL_4:
        if ( result )
        {
          *(_QWORD *)result = 0;
          *(_QWORD *)(result + 8) = 0;
        }
      }
    }
    else
    {
      i = &v7[16 * a3];
      if ( i != (char *)result )
        goto LABEL_4;
    }
    *((_DWORD *)a1 + 4) = a3;
  }
  return result;
}
