// Function: sub_5D4810
// Address: 0x5d4810
//
int __fastcall sub_5D4810(__int64 a1)
{
  char *v2; // rbx
  int v3; // edi
  int result; // eax

  v2 = "*";
  putc(40, stream);
  ++dword_4CF7F40;
  sub_74A390(a1, 1, 0, 0, 0, &qword_4CF7CE0);
  v3 = 32;
  do
  {
    ++v2;
    putc(v3, stream);
    v3 = *(v2 - 1);
  }
  while ( *(v2 - 1) );
  dword_4CF7F40 += 2;
  sub_74D110(a1, 1, 0, &qword_4CF7CE0);
  result = putc(41, stream);
  ++dword_4CF7F40;
  return result;
}
