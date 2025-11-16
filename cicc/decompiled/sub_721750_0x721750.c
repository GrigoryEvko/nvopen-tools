// Function: sub_721750
// Address: 0x721750
//
void __fastcall sub_721750(unsigned __int8 *a1)
{
  _BYTE *v1; // rbx
  int i; // edi

  v1 = a1;
  for ( i = *a1; (_BYTE)i; ++v1 )
  {
    if ( !isalnum(i) )
      *v1 = 95;
    i = (unsigned __int8)v1[1];
  }
}
