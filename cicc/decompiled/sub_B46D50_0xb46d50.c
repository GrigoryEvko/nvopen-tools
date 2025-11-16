// Function: sub_B46D50
// Address: 0xb46d50
//
char __fastcall sub_B46D50(unsigned __int8 *a1)
{
  int v1; // ecx
  unsigned int v2; // ecx
  char result; // al
  __int64 v4; // rdx
  unsigned int v5; // ecx

  v1 = *a1;
  if ( (_BYTE)v1 == 85 )
  {
    v4 = *((_QWORD *)a1 - 4);
    result = 0;
    if ( v4 )
    {
      if ( !*(_BYTE *)v4 && *(_QWORD *)(v4 + 24) == *((_QWORD *)a1 + 10) && (*(_BYTE *)(v4 + 33) & 0x20) != 0 )
      {
        v5 = *(_DWORD *)(v4 + 36);
        if ( v5 <= 0x171 )
        {
          if ( v5 > 0x136 )
          {
            return ((1LL << ((unsigned __int8)v5 - 55)) & 0x7C30000007C0003LL) != 0;
          }
          else if ( v5 > 0xED )
          {
            return v5 - 246 <= 2;
          }
          else
          {
            result = 1;
            if ( v5 <= 0xEA )
              return v5 - 173 <= 1;
          }
        }
      }
    }
  }
  else
  {
    v2 = v1 - 29;
    return v2 <= 0x1E && ((1LL << v2) & 0x70066000) != 0;
  }
  return result;
}
