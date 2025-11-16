// Function: sub_5D5170
// Address: 0x5d5170
//
int __fastcall sub_5D5170(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r13
  int v3; // edi
  char *v4; // rbx
  int result; // eax

  v2 = 0;
  if ( a1 )
    v2 = sub_5D35E0(a1);
  v3 = 99;
  v4 = "har ";
  do
  {
    ++v4;
    putc(v3, stream);
    v3 = *(v4 - 1);
  }
  while ( *(v4 - 1) );
  dword_4CF7F40 += 5;
  ++dword_4CF7F60;
  sub_5D4E40("__nv_no_debug_dummy", 0);
  sub_5D32F0(v2);
  --dword_4CF7F60;
  if ( a2 != 1 )
  {
    putc(91, stream);
    ++dword_4CF7F40;
    sub_5D32F0(a2);
    putc(93, stream);
    ++dword_4CF7F40;
  }
  result = putc(59, stream);
  ++dword_4CF7F40;
  return result;
}
