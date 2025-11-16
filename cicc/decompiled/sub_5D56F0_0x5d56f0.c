// Function: sub_5D56F0
// Address: 0x5d56f0
//
int __fastcall sub_5D56F0(__int64 a1)
{
  __int64 v2; // rdx
  char v3; // al
  int v5; // edi
  char *v6; // rbx
  char *v7; // rbx
  int v8; // edi
  char *v9; // rbx
  int v10; // edi

  v2 = *(_QWORD *)(a1 + 40);
  v3 = *(_BYTE *)(v2 + 28);
  if ( v3 )
  {
    if ( v3 == 3 )
    {
      v9 = ":";
      sub_5D56F0(*(_QWORD *)(v2 + 32));
      v10 = 58;
      do
      {
        ++v9;
        putc(v10, stream);
        v10 = *(v9 - 1);
      }
      while ( *(v9 - 1) );
      dword_4CF7F40 += 2;
    }
    else
    {
      if ( v3 != 6 )
        return sub_5D5580(a1, 0);
      v7 = ":";
      sub_5D56F0(*(_QWORD *)(v2 + 32));
      v8 = 58;
      do
      {
        ++v7;
        putc(v8, stream);
        v8 = *(v7 - 1);
      }
      while ( *(v7 - 1) );
      dword_4CF7F40 += 2;
    }
    return sub_5D5580(a1, 0);
  }
  else
  {
    v5 = 32;
    v6 = "::";
    do
    {
      ++v6;
      putc(v5, stream);
      v5 = *(v6 - 1);
    }
    while ( *(v6 - 1) );
    dword_4CF7F40 += 3;
    return sub_5D5580(a1, 0);
  }
}
