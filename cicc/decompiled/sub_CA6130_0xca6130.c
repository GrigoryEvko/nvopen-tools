// Function: sub_CA6130
// Address: 0xca6130
//
char *__fastcall sub_CA6130(__int64 a1, char *a2)
{
  if ( *(char **)(a1 + 48) == a2 || *a2 == 32 || *a2 == 9 )
    return a2;
  else
    return sub_CA6050(a1, a2);
}
