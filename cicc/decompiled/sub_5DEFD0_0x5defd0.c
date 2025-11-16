// Function: sub_5DEFD0
// Address: 0x5defd0
//
int __fastcall sub_5DEFD0(__int64 a1)
{
  int v2; // edi
  char *v3; // rbx
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  int result; // eax

  v2 = 40;
  v3 = "*(char **)&";
  do
  {
    ++v3;
    putc(v2, stream);
    v2 = *(v3 - 1);
  }
  while ( *(v3 - 1) );
  dword_4CF7F40 += 12;
  sub_5DBFC0(a1, (FILE *)1, v4, v5, v6, v7);
  result = putc(41, stream);
  ++dword_4CF7F40;
  return result;
}
