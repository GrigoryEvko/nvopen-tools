// Function: sub_2B0A810
// Address: 0x2b0a810
//
void __fastcall sub_2B0A810(char *a1, char *a2)
{
  char *v4; // rsi
  int v5; // edi
  char *v6; // rdx
  int v7; // r9d
  char *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdx
  int v11; // ecx
  char *i; // rax

  if ( a1 != a2 )
  {
    v4 = a1 + 8;
    while ( a2 != v4 )
    {
      while ( 1 )
      {
        v5 = *(_DWORD *)v4;
        v6 = v4;
        v4 += 8;
        v7 = *((_DWORD *)v4 - 1);
        if ( v5 >= *(_DWORD *)a1 )
          break;
        v8 = v4;
        v9 = v6 - a1;
        v10 = (v6 - a1) >> 3;
        if ( v9 > 0 )
        {
          do
          {
            v11 = *((_DWORD *)v8 - 4);
            v8 -= 8;
            *(_DWORD *)v8 = v11;
            *((_DWORD *)v8 + 1) = *((_DWORD *)v8 - 1);
            --v10;
          }
          while ( v10 );
        }
        *(_DWORD *)a1 = v5;
        *((_DWORD *)a1 + 1) = v7;
        if ( a2 == v4 )
          return;
      }
      for ( i = v4 - 16; v5 < *(_DWORD *)i; i -= 8 )
      {
        *((_DWORD *)i + 2) = *(_DWORD *)i;
        *((_DWORD *)i + 3) = *((_DWORD *)i + 1);
        v6 = i;
      }
      *(_DWORD *)v6 = v5;
      *((_DWORD *)v6 + 1) = v7;
    }
  }
}
