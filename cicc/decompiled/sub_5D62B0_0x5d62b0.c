// Function: sub_5D62B0
// Address: 0x5d62b0
//
int __fastcall sub_5D62B0(__int64 a1)
{
  int result; // eax
  const char *v3; // r14
  int v4; // eax
  int v5; // edi
  const char *v6; // rbx
  int v7; // r13d

  if ( !qword_4D045BC )
    return sub_5D5A80(a1, 0);
  if ( !a1 )
    return sub_5D5A80(0, 0);
  if ( !*(_QWORD *)(a1 + 8) || (*(_BYTE *)(a1 + 88) & 0x70) != 0x10 )
    return sub_5D5A80(a1, 0);
  ++dword_4CF7F60;
  v3 = (const char *)sub_826B00();
  v4 = strlen(v3);
  v5 = *v3;
  v6 = v3 + 1;
  v7 = v4;
  if ( *v3 )
  {
    do
    {
      ++v6;
      putc(v5, stream);
      v5 = *(v6 - 1);
    }
    while ( *(v6 - 1) );
  }
  dword_4CF7F40 += v7;
  result = sub_5D5A80(a1, 0);
  --dword_4CF7F60;
  return result;
}
