// Function: sub_3946250
// Address: 0x3946250
//
void __fastcall sub_3946250(unsigned int *a1, char *a2, __int64 a3)
{
  char *v3; // rcx
  unsigned int v4; // eax
  char v5; // dl

  v3 = &a2[a3];
  if ( a2 != &a2[a3] )
  {
    v4 = *a1;
    do
    {
      v5 = *a2++;
      v4 = dword_4530980[(unsigned __int8)(v4 ^ v5)] ^ (v4 >> 8);
      *a1 = v4;
    }
    while ( v3 != a2 );
  }
}
