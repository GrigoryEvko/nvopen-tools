// Function: sub_722840
// Address: 0x722840
//
char __fastcall sub_722840(FILE *stream, __int64 a2)
{
  int v2; // eax
  int v3; // eax
  char result; // al
  int v5; // eax
  unsigned __int8 v6; // bl
  int v7; // eax
  unsigned __int64 v8; // rbx
  unsigned __int64 v9; // rax
  int v10; // eax
  unsigned __int8 v11; // r14
  int v12; // eax
  __int64 v13; // rdx
  unsigned __int64 v14; // rdx

  v2 = *(_DWORD *)(a2 + 8);
  if ( v2 )
  {
    v3 = v2 - 1;
    *(_DWORD *)(a2 + 8) = v3;
    return *(_BYTE *)(a2 + v3);
  }
  v5 = getc(stream);
  v6 = v5;
  if ( v5 == -1 )
    return -1;
  v7 = getc(stream);
  if ( v7 == -1 )
    return -1;
  if ( *(_DWORD *)(a2 + 12) == 2 )
    v8 = ((unsigned __int8)v7 << 8) | (unsigned int)v6;
  else
    v8 = (unsigned __int8)v7 | (v6 << 8);
  if ( v8 - 55296 <= 0x7FF )
  {
    if ( v8 - 55296 > 0x3FF )
      return 63;
    v10 = getc(stream);
    v11 = v10;
    if ( v10 != -1 )
    {
      v12 = getc(stream);
      if ( v12 != -1 )
      {
        if ( *(_DWORD *)(a2 + 12) == 2 )
          v13 = ((unsigned __int8)v12 << 8) | (unsigned int)v11;
        else
          v13 = (unsigned __int8)v12 | (v11 << 8);
        if ( (unsigned __int64)(v13 - 56320) <= 0x3FF )
        {
          *(_DWORD *)(a2 + 8) = 3;
          v14 = ((_DWORD)v8 << 10) & 0xFFC00 | (unsigned __int64)(v13 & 0x3FF);
          *(_BYTE *)a2 = v14 & 0x3F | 0x80;
          *(_BYTE *)(a2 + 1) = ((v14 + 0x10000) >> 6) & 0x3F | 0x80;
          v14 += 0x10000LL;
          *(_BYTE *)(a2 + 2) = (v14 >> 12) & 0x3F | 0x80;
          return (v14 >> 18) | 0xF0;
        }
        return 63;
      }
    }
    return -1;
  }
  result = v8;
  if ( v8 > 0x7F )
  {
    v9 = v8 >> 6;
    *(_BYTE *)a2 = v8 & 0x3F | 0x80;
    if ( v8 > 0x7FF )
    {
      *(_DWORD *)(a2 + 8) = 2;
      *(_BYTE *)(a2 + 1) = v9 & 0x3F | 0x80;
      return (v8 >> 12) | 0xE0;
    }
    else
    {
      *(_DWORD *)(a2 + 8) = 1;
      return v9 | 0xC0;
    }
  }
  return result;
}
