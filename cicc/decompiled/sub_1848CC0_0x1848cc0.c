// Function: sub_1848CC0
// Address: 0x1848cc0
//
char __fastcall sub_1848CC0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  char result; // al
  __int64 v4; // rdx
  int v5; // ecx
  __int64 v6; // rdi
  int v7; // esi
  int v8; // r9d
  unsigned int v9; // ecx
  __int64 v10; // r8

  v2 = *a1;
  result = sub_15F3330(a2);
  if ( result )
  {
    if ( *(_BYTE *)(a2 + 16) == 78 )
    {
      v4 = *(_QWORD *)(a2 - 24);
      if ( !*(_BYTE *)(v4 + 16) )
      {
        if ( (*(_BYTE *)(v2 + 8) & 1) != 0 )
        {
          v6 = v2 + 16;
          v7 = 7;
        }
        else
        {
          v5 = *(_DWORD *)(v2 + 24);
          v6 = *(_QWORD *)(v2 + 16);
          if ( !v5 )
            return result;
          v7 = v5 - 1;
        }
        v8 = 1;
        v9 = v7 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v10 = *(_QWORD *)(v6 + 8LL * v9);
        if ( v4 == v10 )
        {
          return 0;
        }
        else
        {
          while ( v10 != -8 )
          {
            v9 = v7 & (v8 + v9);
            v10 = *(_QWORD *)(v6 + 8LL * v9);
            if ( v4 == v10 )
              return 0;
            ++v8;
          }
        }
      }
    }
  }
  return result;
}
