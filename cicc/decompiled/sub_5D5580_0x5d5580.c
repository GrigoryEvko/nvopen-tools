// Function: sub_5D5580
// Address: 0x5d5580
//
int __fastcall sub_5D5580(__int64 a1, int a2)
{
  const char *v3; // r13
  char *v4; // rdi
  const char *v5; // r12
  const char *v6; // rbx
  int result; // eax
  int v8; // edi
  int v9; // r13d
  const char *v10; // rbx
  int v11; // edi
  int v12; // r12d
  int v13; // edi
  char *v14; // rbx
  int v15; // edi
  char *v16; // rbx

  v3 = *(const char **)(a1 + 8);
  if ( (*(_BYTE *)(a1 + 89) & 8) != 0 )
  {
    v4 = *(char **)(a1 + 16);
    if ( !v4 )
      v4 = (char *)v3;
    v5 = (const char *)sub_8257B0(v4);
    if ( a2 )
    {
      v13 = 58;
      v14 = ":";
      do
      {
        ++v14;
        putc(v13, stream);
        v13 = *(v14 - 1);
      }
      while ( *(v14 - 1) );
      dword_4CF7F40 += 2;
    }
    v6 = v5 + 1;
    result = strlen(v5);
    v8 = *v5;
    v9 = result;
    if ( *v5 )
    {
      do
      {
        ++v6;
        result = putc(v8, stream);
        v8 = *(v6 - 1);
      }
      while ( *(v6 - 1) );
    }
    dword_4CF7F40 += v9;
  }
  else if ( v3 )
  {
    if ( a2 )
    {
      v15 = 58;
      v16 = ":";
      do
      {
        ++v16;
        putc(v15, stream);
        v15 = *(v16 - 1);
      }
      while ( *(v16 - 1) );
      dword_4CF7F40 += 2;
      v3 = *(const char **)(a1 + 8);
    }
    v10 = v3 + 1;
    result = strlen(v3);
    v11 = *v3;
    v12 = result;
    if ( *v3 )
    {
      do
      {
        ++v10;
        result = putc(v11, stream);
        v11 = *(v10 - 1);
      }
      while ( *(v10 - 1) );
    }
    dword_4CF7F40 += v12;
  }
  else
  {
    return sub_5D34A0();
  }
  return result;
}
