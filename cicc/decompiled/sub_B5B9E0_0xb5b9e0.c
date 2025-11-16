// Function: sub_B5B9E0
// Address: 0xb5b9e0
//
void sub_B5B9E0()
{
  char *v0; // rdi
  char v1; // al

  v0 = (char *)&unk_4BB0900;
  if ( &unk_4BB0900 != (_UNKNOWN *)a2000 )
  {
    v1 = 0;
    do
    {
      *v0++ ^= v1;
      v1 += 3;
    }
    while ( a2000 != v0 );
  }
}
