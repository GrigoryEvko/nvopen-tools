// Function: sub_5D71E0
// Address: 0x5d71e0
//
int __fastcall sub_5D71E0(__int64 a1)
{
  __int64 v2; // rax
  char v3; // al
  int v4; // edi
  char *v5; // rbx
  int result; // eax
  int v7; // edi
  const char *v8; // r12

  if ( (*(_BYTE *)(a1 + 143) & 0x10) != 0 )
  {
    *(_BYTE *)(a1 + 88) = *(_BYTE *)(a1 + 88) & 0x8F | 0x30;
    if ( dword_4F068D4 )
    {
      v7 = 95;
      v8 = "_builtin_va_list";
      do
      {
        ++v8;
        result = putc(v7, stream);
        v7 = *(v8 - 1);
      }
      while ( *(v8 - 1) );
      dword_4CF7F40 += 17;
      return result;
    }
    *(_QWORD *)(a1 + 8) = "va_list";
  }
  if ( dword_4CF7EFC )
  {
    v2 = *(_QWORD *)(a1 + 40);
    if ( v2 )
    {
      if ( !*(_BYTE *)(v2 + 28) )
      {
        v3 = *(_BYTE *)(a1 + 89);
        if ( (v3 & 2) == 0
          && (*(_BYTE *)(a1 + 141) & 8) == 0
          && *(_QWORD *)(a1 + 8)
          && (v3 & 8) == 0
          && (*(_BYTE *)(a1 + 140) != 12 || *(char *)(a1 + 185) >= 0) )
        {
          v4 = 58;
          v5 = ":";
          do
          {
            ++v5;
            putc(v4, stream);
            v4 = *(v5 - 1);
          }
          while ( *(v5 - 1) );
          dword_4CF7F40 += 2;
        }
      }
    }
  }
  return sub_5D5A80(a1, 0);
}
