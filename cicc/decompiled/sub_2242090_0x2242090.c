// Function: sub_2242090
// Address: 0x2242090
//
unsigned int __fastcall sub_2242090(pthread_t *a1)
{
  pthread_t v2; // rdi
  unsigned int result; // eax

  v2 = *a1;
  if ( !v2 )
  {
    result = 22;
LABEL_5:
    sub_4264C5(result);
  }
  result = pthread_join(v2, 0);
  if ( result )
    goto LABEL_5;
  *a1 = 0;
  return result;
}
