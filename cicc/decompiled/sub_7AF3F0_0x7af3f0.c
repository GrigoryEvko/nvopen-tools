// Function: sub_7AF3F0
// Address: 0x7af3f0
//
int __fastcall sub_7AF3F0(char a1)
{
  fprintf(
    qword_4D04908,
    "L %lu \"%s\"",
    (unsigned int)(*(_DWORD *)(unk_4F064B0 + 40LL) + 1),
    *(const char **)(unk_4F064B0 + 8LL));
  if ( a1 != 32 )
  {
    putc(32, qword_4D04908);
    putc(a1, qword_4D04908);
  }
  return putc(10, qword_4D04908);
}
