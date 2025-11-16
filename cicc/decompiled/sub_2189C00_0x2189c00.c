// Function: sub_2189C00
// Address: 0x2189c00
//
char __fastcall sub_2189C00(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, const char *a5)
{
  bool v5; // zf
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  _DWORD *v9; // rdx

  v5 = strcmp(a5, "ftz") == 0;
  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  LOBYTE(v7) = !v5;
  if ( v5 )
  {
    if ( (v6 & 0x100) != 0 )
    {
      v9 = *(_DWORD **)(a4 + 24);
      v7 = *(_QWORD *)(a4 + 16) - (_QWORD)v9;
      if ( v7 <= 3 )
      {
        LOBYTE(v7) = sub_16E7EE0(a4, ".ftz", 4u);
      }
      else
      {
        *v9 = 2054448686;
        *(_QWORD *)(a4 + 24) += 4LL;
      }
    }
  }
  else
  {
    LODWORD(v7) = (unsigned __int8)v6;
    v8 = *(_QWORD *)(a4 + 24);
    switch ( (int)v7 )
    {
      case 0:
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 16) - v8) <= 2 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".eq", 3u);
        }
        else
        {
          *(_BYTE *)(v8 + 2) = 113;
          *(_WORD *)v8 = 25902;
          *(_QWORD *)(a4 + 24) += 3LL;
          LOBYTE(v7) = 46;
        }
        break;
      case 1:
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 16) - v8) <= 2 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".ne", 3u);
        }
        else
        {
          *(_BYTE *)(v8 + 2) = 101;
          *(_WORD *)v8 = 28206;
          *(_QWORD *)(a4 + 24) += 3LL;
          LOBYTE(v7) = 46;
        }
        break;
      case 2:
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 16) - v8) <= 2 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".lt", 3u);
        }
        else
        {
          *(_BYTE *)(v8 + 2) = 116;
          *(_WORD *)v8 = 27694;
          *(_QWORD *)(a4 + 24) += 3LL;
          LOBYTE(v7) = 46;
        }
        break;
      case 3:
        v7 = *(_QWORD *)(a4 + 16) - v8;
        if ( v7 <= 2 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".le", 3u);
        }
        else
        {
          *(_BYTE *)(v8 + 2) = 101;
          *(_WORD *)v8 = 27694;
          *(_QWORD *)(a4 + 24) += 3LL;
        }
        break;
      case 4:
        v7 = *(_QWORD *)(a4 + 16) - v8;
        if ( v7 <= 2 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".gt", 3u);
        }
        else
        {
          *(_BYTE *)(v8 + 2) = 116;
          *(_WORD *)v8 = 26414;
          *(_QWORD *)(a4 + 24) += 3LL;
        }
        break;
      case 5:
        v7 = *(_QWORD *)(a4 + 16) - v8;
        if ( v7 <= 2 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".ge", 3u);
        }
        else
        {
          *(_BYTE *)(v8 + 2) = 101;
          *(_WORD *)v8 = 26414;
          *(_QWORD *)(a4 + 24) += 3LL;
        }
        break;
      case 6:
        v7 = *(_QWORD *)(a4 + 16) - v8;
        if ( v7 <= 2 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".lo", 3u);
        }
        else
        {
          *(_BYTE *)(v8 + 2) = 111;
          *(_WORD *)v8 = 27694;
          *(_QWORD *)(a4 + 24) += 3LL;
        }
        break;
      case 7:
        v7 = *(_QWORD *)(a4 + 16) - v8;
        if ( v7 <= 2 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, asc_432C6C4, 3u);
        }
        else
        {
          *(_BYTE *)(v8 + 2) = 115;
          *(_WORD *)v8 = 27694;
          *(_QWORD *)(a4 + 24) += 3LL;
        }
        break;
      case 8:
        v7 = *(_QWORD *)(a4 + 16) - v8;
        if ( v7 <= 2 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".hi", 3u);
        }
        else
        {
          *(_BYTE *)(v8 + 2) = 105;
          *(_WORD *)v8 = 26670;
          *(_QWORD *)(a4 + 24) += 3LL;
        }
        break;
      case 9:
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 16) - v8) <= 2 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".hs", 3u);
        }
        else
        {
          *(_BYTE *)(v8 + 2) = 115;
          *(_WORD *)v8 = 26670;
          *(_QWORD *)(a4 + 24) += 3LL;
          LOBYTE(v7) = 46;
        }
        break;
      case 10:
        v7 = *(_QWORD *)(a4 + 16) - v8;
        if ( v7 <= 3 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".equ", 4u);
        }
        else
        {
          *(_DWORD *)v8 = 1970365742;
          *(_QWORD *)(a4 + 24) += 4LL;
        }
        break;
      case 11:
        v7 = *(_QWORD *)(a4 + 16) - v8;
        if ( v7 <= 3 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".neu", 4u);
        }
        else
        {
          *(_DWORD *)v8 = 1969581614;
          *(_QWORD *)(a4 + 24) += 4LL;
        }
        break;
      case 12:
        v7 = *(_QWORD *)(a4 + 16) - v8;
        if ( v7 <= 3 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".ltu", 4u);
        }
        else
        {
          *(_DWORD *)v8 = 1970564142;
          *(_QWORD *)(a4 + 24) += 4LL;
        }
        break;
      case 13:
        v7 = *(_QWORD *)(a4 + 16) - v8;
        if ( v7 <= 3 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".leu", 4u);
        }
        else
        {
          *(_DWORD *)v8 = 1969581102;
          *(_QWORD *)(a4 + 24) += 4LL;
        }
        break;
      case 14:
        v7 = *(_QWORD *)(a4 + 16) - v8;
        if ( v7 <= 3 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".gtu", 4u);
        }
        else
        {
          *(_DWORD *)v8 = 1970562862;
          *(_QWORD *)(a4 + 24) += 4LL;
        }
        break;
      case 15:
        v7 = *(_QWORD *)(a4 + 16) - v8;
        if ( v7 <= 3 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".geu", 4u);
        }
        else
        {
          *(_DWORD *)v8 = 1969579822;
          *(_QWORD *)(a4 + 24) += 4LL;
        }
        break;
      case 16:
        v7 = *(_QWORD *)(a4 + 16) - v8;
        if ( v7 <= 3 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".num", 4u);
        }
        else
        {
          *(_DWORD *)v8 = 1836412462;
          *(_QWORD *)(a4 + 24) += 4LL;
        }
        break;
      case 17:
        v7 = *(_QWORD *)(a4 + 16) - v8;
        if ( v7 <= 3 )
        {
          LOBYTE(v7) = sub_16E7EE0(a4, ".nan", 4u);
        }
        else
        {
          *(_DWORD *)v8 = 1851878958;
          *(_QWORD *)(a4 + 24) += 4LL;
        }
        break;
      default:
        return v7;
    }
  }
  return v7;
}
