// Function: sub_5D4790
// Address: 0x5d4790
//
int __fastcall sub_5D4790(_DWORD *a1)
{
  int v1; // edx
  int result; // eax
  int v5; // edi
  char *v6; // r12

  v1 = a1[3];
  *a1 = 1;
  if ( !v1 )
  {
    v5 = 32;
    v6 = "= ";
    do
    {
      ++v6;
      putc(v5, stream);
      v5 = *(v6 - 1);
    }
    while ( *(v6 - 1) );
    dword_4CF7F40 += 3;
  }
  result = a1[1];
  if ( result )
  {
    do
    {
      result = putc(123, stream);
      ++dword_4CF7F40;
    }
    while ( a1[1]-- != 1 );
  }
  return result;
}
