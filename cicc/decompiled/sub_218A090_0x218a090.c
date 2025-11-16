// Function: sub_218A090
// Address: 0x218a090
//
char __fastcall sub_218A090(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, const char *a5)
{
  bool v5; // zf
  __int64 v6; // rdx
  _BYTE *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx

  v5 = strcmp(a5, "volatile") == 0;
  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  LOBYTE(v7) = !v5;
  if ( v5 )
  {
    if ( (_DWORD)v6 )
    {
      v8 = *(_QWORD *)(a4 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 16) - v8) <= 8 )
      {
        LOBYTE(v7) = sub_16E7EE0(a4, ".volatile", 9u);
      }
      else
      {
        *(_BYTE *)(v8 + 8) = 101;
        *(_QWORD *)v8 = 0x6C6974616C6F762ELL;
        *(_QWORD *)(a4 + 24) += 9LL;
        LOBYTE(v7) = 46;
      }
    }
  }
  else
  {
    v5 = strcmp(a5, "addsp") == 0;
    LOBYTE(v7) = !v5;
    if ( v5 )
    {
      switch ( (int)v6 )
      {
        case 0:
          return (char)v7;
        case 1:
          v13 = *(_QWORD *)(a4 + 24);
          v7 = (_BYTE *)(*(_QWORD *)(a4 + 16) - v13);
          if ( (unsigned __int64)v7 <= 6 )
          {
            LOBYTE(v7) = sub_16E7EE0(a4, ".global", 7u);
          }
          else
          {
            *(_DWORD *)v13 = 1869375278;
            *(_WORD *)(v13 + 4) = 24930;
            *(_BYTE *)(v13 + 6) = 108;
            *(_QWORD *)(a4 + 24) += 7LL;
          }
          break;
        case 2:
          v14 = *(_QWORD *)(a4 + 24);
          v7 = (_BYTE *)(*(_QWORD *)(a4 + 16) - v14);
          if ( (unsigned __int64)v7 <= 5 )
          {
            LOBYTE(v7) = sub_16E7EE0(a4, ".const", 6u);
          }
          else
          {
            *(_DWORD *)v14 = 1852793646;
            *(_WORD *)(v14 + 4) = 29811;
            *(_QWORD *)(a4 + 24) += 6LL;
          }
          break;
        case 3:
          v10 = *(_QWORD *)(a4 + 24);
          v7 = (_BYTE *)(*(_QWORD *)(a4 + 16) - v10);
          if ( (unsigned __int64)v7 <= 6 )
          {
            LOBYTE(v7) = sub_16E7EE0(a4, ".shared", 7u);
          }
          else
          {
            *(_DWORD *)v10 = 1634235182;
            *(_WORD *)(v10 + 4) = 25970;
            *(_BYTE *)(v10 + 6) = 100;
            *(_QWORD *)(a4 + 24) += 7LL;
          }
          break;
        case 4:
          v12 = *(_QWORD *)(a4 + 24);
          v7 = (_BYTE *)(*(_QWORD *)(a4 + 16) - v12);
          if ( (unsigned __int64)v7 <= 5 )
          {
            LOBYTE(v7) = sub_16E7EE0(a4, ".param", 6u);
          }
          else
          {
            *(_DWORD *)v12 = 1918988334;
            *(_WORD *)(v12 + 4) = 28001;
            *(_QWORD *)(a4 + 24) += 6LL;
          }
          break;
        case 5:
          v11 = *(_QWORD *)(a4 + 24);
          v7 = (_BYTE *)(*(_QWORD *)(a4 + 16) - v11);
          if ( (unsigned __int64)v7 <= 5 )
          {
            LOBYTE(v7) = sub_16E7EE0(a4, ".local", 6u);
          }
          else
          {
            *(_DWORD *)v11 = 1668246574;
            *(_WORD *)(v11 + 4) = 27745;
            *(_QWORD *)(a4 + 24) += 6LL;
          }
          break;
      }
    }
    else
    {
      v5 = strcmp(a5, "sign") == 0;
      LOBYTE(v7) = !v5;
      if ( v5 )
      {
        v7 = *(_BYTE **)(a4 + 24);
        if ( (_DWORD)v6 == 1 )
        {
          if ( *(_BYTE **)(a4 + 16) == v7 )
          {
            LOBYTE(v7) = sub_16E7EE0(a4, "s", 1u);
          }
          else
          {
            *v7 = 115;
            ++*(_QWORD *)(a4 + 24);
          }
        }
        else if ( (_DWORD)v6 )
        {
          if ( (_DWORD)v6 == 3 )
          {
            if ( *(_BYTE **)(a4 + 16) == v7 )
            {
              LOBYTE(v7) = sub_16E7EE0(a4, "b", 1u);
            }
            else
            {
              *v7 = 98;
              ++*(_QWORD *)(a4 + 24);
            }
          }
          else if ( *(_BYTE **)(a4 + 16) == v7 )
          {
            LOBYTE(v7) = sub_16E7EE0(a4, "f", 1u);
          }
          else
          {
            *v7 = 102;
            ++*(_QWORD *)(a4 + 24);
          }
        }
        else if ( *(_BYTE **)(a4 + 16) == v7 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, (char *)"u", 1u);
        }
        else
        {
          *v7 = 117;
          ++*(_QWORD *)(a4 + 24);
        }
      }
      else if ( (_DWORD)v6 == 2 )
      {
        v15 = *(_QWORD *)(a4 + 24);
        v7 = (_BYTE *)(*(_QWORD *)(a4 + 16) - v15);
        if ( (unsigned __int64)v7 <= 2 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".v2", 3u);
        }
        else
        {
          *(_BYTE *)(v15 + 2) = 50;
          *(_WORD *)v15 = 30254;
          *(_QWORD *)(a4 + 24) += 3LL;
        }
      }
      else if ( (_DWORD)v6 == 4 )
      {
        v9 = *(_QWORD *)(a4 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 16) - v9) <= 2 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".v4", 3u);
        }
        else
        {
          *(_BYTE *)(v9 + 2) = 52;
          *(_WORD *)v9 = 30254;
          *(_QWORD *)(a4 + 24) += 3LL;
          LOBYTE(v7) = 46;
        }
      }
    }
  }
  return (char)v7;
}
