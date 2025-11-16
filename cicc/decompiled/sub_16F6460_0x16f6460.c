// Function: sub_16F6460
// Address: 0x16f6460
//
char *__fastcall sub_16F6460(__int64 a1, char *a2)
{
  if ( *(char **)(a1 + 48) == a2 || *a2 == 32 || *a2 == 9 )
    return a2;
  else
    return sub_16F6380(a1, a2);
}
