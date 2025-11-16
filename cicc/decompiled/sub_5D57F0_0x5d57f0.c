// Function: sub_5D57F0
// Address: 0x5d57f0
//
void __fastcall sub_5D57F0(__int64 a1)
{
  __int64 v2; // rdi
  int v4; // edi
  char *v5; // rbx
  int v6; // edi
  const char *v7; // rbx
  int v8; // edi
  const char *v9; // rbx

  v2 = *(unsigned __int8 *)(a1 + 136);
  if ( (_BYTE)v2 == 5 )
  {
    if ( (unsigned int)sub_8D3B80(*(_QWORD *)(a1 + 120)) || (*(_BYTE *)(a1 + 169) & 0x40) != 0 )
    {
      if ( dword_4CF7EA0 )
      {
        if ( !dword_4CF7EA4++ )
        {
          v6 = 47;
          v7 = "*";
          do
          {
            ++v7;
            putc(v6, stream);
            v6 = *(v7 - 1);
            ++dword_4CF7F40;
          }
          while ( (_BYTE)v6 );
        }
        v4 = 114;
        v5 = "egister";
        do
        {
          ++v5;
          putc(v4, stream);
          v4 = *(v5 - 1);
        }
        while ( *(v5 - 1) );
        dword_4CF7F40 += 8;
        if ( !--dword_4CF7EA4 )
        {
          v8 = 42;
          v9 = "/";
          do
          {
            ++v9;
            putc(v8, stream);
            v8 = *(v9 - 1);
            ++dword_4CF7F40;
          }
          while ( (_BYTE)v8 );
        }
        putc(32, stream);
        ++dword_4CF7F40;
      }
    }
    else
    {
      sub_5D3F60(*(unsigned __int8 *)(a1 + 136));
    }
  }
  else
  {
    sub_5D3F60(v2);
  }
}
