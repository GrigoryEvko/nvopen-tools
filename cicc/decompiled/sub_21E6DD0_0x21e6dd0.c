// Function: sub_21E6DD0
// Address: 0x21e6dd0
//
size_t __fastcall sub_21E6DD0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r15
  size_t result; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdx
  _QWORD *v13; // rdx
  _WORD *v14; // rdx
  __int64 v15; // rdx
  void *v16; // rdx
  size_t v17; // rdx
  char *v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rdx

  v8 = sub_1C278B0(*(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8));
  result = strlen((const char *)a5);
  switch ( result )
  {
    case 3uLL:
      if ( *(_WORD *)a5 == 28534 && *(_BYTE *)(a5 + 2) == 108 && (v8 & 0x200) != 0 )
      {
        v12 = *(_QWORD *)(a4 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 16) - v12) > 8 )
        {
          *(_BYTE *)(v12 + 8) = 101;
          *(_QWORD *)v12 = 0x6C6974616C6F762ELL;
          *(_QWORD *)(a4 + 24) += 9LL;
          return 0x6C6974616C6F762ELL;
        }
        v17 = 9;
        v18 = ".volatile";
        return sub_16E7EE0(a4, v18, v17);
      }
      break;
    case 2uLL:
      if ( *(_WORD *)a5 == 29555 )
      {
        result = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * (a3 + 1) + 8);
        if ( (_DWORD)result == 3 )
        {
          v20 = *(_QWORD *)(a4 + 24);
          result = *(_QWORD *)(a4 + 16) - v20;
          if ( result > 6 )
          {
            *(_DWORD *)v20 = 1634235182;
            *(_WORD *)(v20 + 4) = 25970;
            *(_BYTE *)(v20 + 6) = 100;
            *(_QWORD *)(a4 + 24) += 7LL;
            return result;
          }
          v17 = 7;
          v18 = ".shared";
        }
        else if ( (int)result > 3 )
        {
          if ( (_DWORD)result != 5 )
          {
            v15 = *(_QWORD *)(a4 + 24);
            result = *(_QWORD *)(a4 + 16) - v15;
            if ( result > 5 )
            {
              *(_DWORD *)v15 = 1918988334;
              *(_WORD *)(v15 + 4) = 28001;
              *(_QWORD *)(a4 + 24) += 6LL;
              return result;
            }
            v17 = 6;
            v18 = ".param";
            return sub_16E7EE0(a4, v18, v17);
          }
          v19 = *(_QWORD *)(a4 + 24);
          result = *(_QWORD *)(a4 + 16) - v19;
          if ( result > 5 )
          {
            *(_DWORD *)v19 = 1668246574;
            *(_WORD *)(v19 + 4) = 27745;
            *(_QWORD *)(a4 + 24) += 6LL;
            return result;
          }
          v17 = 6;
          v18 = ".local";
        }
        else
        {
          if ( !(_DWORD)result )
            return result;
          v10 = *(_QWORD *)(a4 + 24);
          result = *(_QWORD *)(a4 + 16) - v10;
          if ( result > 6 )
          {
            *(_DWORD *)v10 = 1869375278;
            *(_WORD *)(v10 + 4) = 24930;
            *(_BYTE *)(v10 + 6) = 108;
            *(_QWORD *)(a4 + 24) += 7LL;
            return result;
          }
          v17 = 7;
          v18 = ".global";
        }
      }
      else
      {
        if ( (v8 & 0x100) == 0 )
          return result;
        v11 = *(_QWORD *)(a4 + 24);
        result = *(_QWORD *)(a4 + 16) - v11;
        if ( result > 2 )
        {
          *(_BYTE *)(v11 + 2) = 99;
          *(_WORD *)v11 = 28206;
          *(_QWORD *)(a4 + 24) += 3LL;
          return result;
        }
        v17 = 3;
        v18 = (char *)&unk_435F0B4;
      }
      return sub_16E7EE0(a4, v18, v17);
    case 7uLL:
      if ( *(_DWORD *)a5 == 1668506980 && *(_WORD *)(a5 + 4) == 30067 && *(_BYTE *)(a5 + 6) == 102 )
      {
        if ( (v8 & 0x400) != 0 )
        {
          v16 = *(void **)(a4 + 24);
          if ( *(_QWORD *)(a4 + 16) - (_QWORD)v16 > 0xEu )
          {
            qmemcpy(v16, ".L2::cache_hint", 15);
            *(_QWORD *)(a4 + 24) += 15LL;
            return 0x6361633A3A324C2ELL;
          }
          v17 = 15;
          v18 = ".L2::cache_hint";
          return sub_16E7EE0(a4, v18, v17);
        }
      }
      else if ( (v8 & 0x1000000000LL) != 0 )
      {
        v13 = *(_QWORD **)(a4 + 24);
        if ( *(_QWORD *)(a4 + 16) - (_QWORD)v13 > 7u )
        {
          *v13 = 0x64656966696E752ELL;
          *(_QWORD *)(a4 + 24) += 8LL;
          return 0x64656966696E752ELL;
        }
        v17 = 8;
        v18 = ".unified";
        return sub_16E7EE0(a4, v18, v17);
      }
      break;
    default:
      if ( (v8 & 0x400) != 0 )
      {
        v14 = *(_WORD **)(a4 + 24);
        if ( *(_QWORD *)(a4 + 16) - (_QWORD)v14 <= 1u )
        {
          sub_16E7EE0(a4, ", ", 2u);
        }
        else
        {
          *v14 = 8236;
          *(_QWORD *)(a4 + 24) += 2LL;
        }
        return (size_t)sub_21897A0(a1, a2, *(_DWORD *)(a2 + 24) - 1, a4);
      }
      break;
  }
  return result;
}
