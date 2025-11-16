// Function: sub_BC7570
// Address: 0xbc7570
//
char *__fastcall sub_BC7570(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 v2; // rax
  char *result; // rax
  char *v4; // r13
  char *v5; // rbx
  char *v6; // rsi

  v1 = a1[24];
  if ( v1 != a1[25] )
    a1[25] = v1;
  v2 = a1[17];
  if ( v2 != a1[18] )
    a1[18] = v2;
  result = (char *)a1[20];
  v4 = (char *)a1[21];
  v5 = result + 8;
  if ( result != v4 )
  {
    while ( 1 )
    {
      v6 = (char *)a1[18];
      if ( v6 == (char *)a1[19] )
      {
        sub_931B30((__int64)(a1 + 17), v6, v5);
        result = v5 + 16;
        if ( v4 == v5 + 8 )
          return result;
      }
      else
      {
        if ( v6 )
        {
          *v6 = *v5;
          v6 = (char *)a1[18];
        }
        result = v5 + 16;
        a1[18] = v6 + 1;
        if ( v4 == v5 + 8 )
          return result;
      }
      v5 = result;
    }
  }
  return result;
}
