// Function: sub_1098F50
// Address: 0x1098f50
//
__int64 __fastcall sub_1098F50(unsigned int a1, char *a2, __int64 a3)
{
  char *v3; // rcx
  unsigned int v4; // eax
  char v5; // dl

  v3 = &a2[a3];
  v4 = ~a1;
  if ( a2 == &a2[a3] )
    return a1;
  do
  {
    v5 = *a2++;
    v4 = dword_3F90200[(unsigned __int8)(v4 ^ v5)] ^ (v4 >> 8);
  }
  while ( v3 != a2 );
  return ~v4;
}
