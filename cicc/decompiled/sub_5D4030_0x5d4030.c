// Function: sub_5D4030
// Address: 0x5d4030
//
int __fastcall sub_5D4030(__int64 a1, const char *a2)
{
  char *v3; // rbx
  int v4; // edi
  __int64 v5; // r13
  const char *v6; // r15
  int v7; // eax
  int v8; // edi
  int v9; // ebx
  int result; // eax

  v3 = (char *)&unk_39FBEB5;
  putc(40, stream);
  ++dword_4CF7F40;
  sub_74A390(a1, 0, 0, 0, 0, &qword_4CF7CE0);
  sub_74D110(a1, 0, 0, &qword_4CF7CE0);
  v4 = 41;
  do
  {
    ++v3;
    putc(v4, stream);
    v4 = *(v3 - 1);
  }
  while ( *(v3 - 1) );
  dword_4CF7F40 += 2;
  v5 = sub_8D4620(a1);
  if ( v5 )
  {
    while ( 1 )
    {
      v6 = a2 + 1;
      v7 = strlen(a2);
      v8 = *a2;
      v9 = v7;
      if ( *a2 )
      {
        do
        {
          ++v6;
          putc(v8, stream);
          v8 = *(v6 - 1);
        }
        while ( *(v6 - 1) );
      }
      dword_4CF7F40 += v9;
      if ( v5 == 1 )
        break;
      --v5;
      putc(44, stream);
      ++dword_4CF7F40;
    }
  }
  result = putc(125, stream);
  ++dword_4CF7F40;
  return result;
}
