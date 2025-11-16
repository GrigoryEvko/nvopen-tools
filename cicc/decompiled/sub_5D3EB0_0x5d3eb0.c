// Function: sub_5D3EB0
// Address: 0x5d3eb0
//
__int64 __fastcall sub_5D3EB0(char *a1)
{
  int v2; // r12d
  int v3; // eax
  int v4; // edi
  char *v5; // rbx
  __int64 result; // rax

  v2 = dword_4CF7F38;
  if ( dword_4CF7F40 )
    sub_5D37C0();
  v3 = dword_4CF7F60;
  v4 = *a1;
  v5 = a1 + 1;
  dword_4CF7F38 = 0;
  ++dword_4CF7F60;
  if ( (_BYTE)v4 )
  {
    do
    {
      ++v5;
      putc(v4, stream);
      v4 = *(v5 - 1);
      ++dword_4CF7F40;
    }
    while ( (_BYTE)v4 );
    v3 = dword_4CF7F60 - 1;
  }
  dword_4CF7F60 = v3;
  result = sub_5D37C0();
  dword_4CF7F38 = v2;
  return result;
}
