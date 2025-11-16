// Function: sub_720F70
// Address: 0x720f70
//
int __fastcall sub_720F70(FILE **a1)
{
  FILE *v2; // rdi
  int result; // eax

  v2 = *a1;
  if ( v2 )
  {
    result = fclose(v2);
    *a1 = 0;
  }
  return result;
}
