// Function: sub_1DCCC40
// Address: 0x1dccc40
//
char *__fastcall sub_1DCCC40(char *a1, int a2, __int64 a3)
{
  char *result; // rax
  _BYTE *v5; // rsi
  __int64 v6; // [rsp+8h] [rbp-18h] BYREF

  result = sub_1DCC790(a1, a2);
  if ( *((char **)result + 1) == result + 8 )
  {
    v6 = a3;
    v5 = (_BYTE *)*((_QWORD *)result + 5);
    if ( v5 == *((_BYTE **)result + 6) )
    {
      return sub_1DCC370((__int64)(result + 32), v5, &v6);
    }
    else
    {
      if ( v5 )
      {
        *(_QWORD *)v5 = a3;
        v5 = (_BYTE *)*((_QWORD *)result + 5);
      }
      *((_QWORD *)result + 5) = v5 + 8;
    }
  }
  return result;
}
