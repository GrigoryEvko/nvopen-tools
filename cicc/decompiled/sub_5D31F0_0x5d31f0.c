// Function: sub_5D31F0
// Address: 0x5d31f0
//
int __fastcall sub_5D31F0(const char *a1)
{
  int result; // eax
  int v3; // edi
  int v4; // r13d
  const char *v5; // rbx

  result = strlen(a1);
  v3 = *a1;
  v4 = result;
  if ( *a1 )
  {
    v5 = a1 + 1;
    do
    {
      ++v5;
      result = putc(v3, stream);
      v3 = *(v5 - 1);
    }
    while ( *(v5 - 1) );
  }
  dword_4CF7F40 += v4;
  return result;
}
