// Function: sub_5D3190
// Address: 0x5d3190
//
int __fastcall sub_5D3190(char *a1)
{
  char *v1; // rbx
  int i; // edi
  int result; // eax

  v1 = a1 + 1;
  for ( i = *a1; (_BYTE)i; ++dword_4CF7F40 )
  {
    ++v1;
    result = putc(i, stream);
    i = *(v1 - 1);
  }
  return result;
}
