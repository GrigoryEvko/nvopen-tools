// Function: sub_21E7FE0
// Address: 0x21e7fe0
//
char __fastcall sub_21E7FE0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  unsigned __int64 v6; // rax
  __int64 v7; // r9
  __int64 v8; // rdx
  bool v9; // zf
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  void *v16; // rdx
  char *v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rdx
  char *v20; // rax
  bool v21; // cf
  char *v22; // rax
  char *v23; // rax
  char *v24; // rax
  char *v25; // rax

  LOBYTE(v6) = a4;
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL * a2 + 8);
  if ( a4 )
  {
    if ( !strcmp((const char *)a4, "addsp") )
    {
      if ( (_DWORD)v7 == 3 )
      {
        v15 = *(_QWORD *)(a3 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v15) <= 6 )
        {
          LOBYTE(v6) = sub_16E7EE0(a3, ".shared", 7u);
        }
        else
        {
          *(_DWORD *)v15 = 1634235182;
          *(_WORD *)(v15 + 4) = 25970;
          *(_BYTE *)(v15 + 6) = 100;
          *(_QWORD *)(a3 + 24) += 7LL;
          LOBYTE(v6) = 114;
        }
      }
      else if ( (int)v7 > 3 )
      {
        v14 = *(_QWORD *)(a3 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v14) <= 5 )
        {
          LOBYTE(v6) = sub_16E7EE0(a3, ".local", 6u);
        }
        else
        {
          *(_DWORD *)v14 = 1668246574;
          *(_WORD *)(v14 + 4) = 27745;
          *(_QWORD *)(a3 + 24) += 6LL;
          LOBYTE(v6) = 97;
        }
      }
      else if ( (_DWORD)v7 )
      {
        v8 = *(_QWORD *)(a3 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v8) <= 6 )
        {
          LOBYTE(v6) = sub_16E7EE0(a3, ".global", 7u);
        }
        else
        {
          *(_DWORD *)v8 = 1869375278;
          *(_WORD *)(v8 + 4) = 24930;
          *(_BYTE *)(v8 + 6) = 108;
          *(_QWORD *)(a3 + 24) += 7LL;
          LOBYTE(v6) = 98;
        }
      }
    }
    else if ( *(_BYTE *)a4 == 97 && *(_BYTE *)(a4 + 1) == 98 && !*(_BYTE *)(a4 + 2) )
    {
      v10 = *(_QWORD *)(a3 + 16);
      v6 = *(_QWORD *)(a3 + 24);
      if ( (_DWORD)v7 )
      {
        if ( v10 == v6 )
        {
          LOBYTE(v6) = sub_16E7EE0(a3, "b", 1u);
        }
        else
        {
          *(_BYTE *)v6 = 98;
          ++*(_QWORD *)(a3 + 24);
        }
      }
      else if ( v10 == v6 )
      {
        LOBYTE(v6) = sub_16E7EE0(a3, "a", 1u);
      }
      else
      {
        *(_BYTE *)v6 = 97;
        ++*(_QWORD *)(a3 + 24);
      }
    }
    else if ( !strcmp((const char *)a4, "rowcol") )
    {
      v11 = *(_QWORD *)(a3 + 24);
      v6 = *(_QWORD *)(a3 + 16) - v11;
      if ( (_DWORD)v7 )
      {
        if ( v6 <= 2 )
        {
          LOBYTE(v6) = sub_16E7EE0(a3, "col", 3u);
        }
        else
        {
          *(_BYTE *)(v11 + 2) = 108;
          *(_WORD *)v11 = 28515;
          *(_QWORD *)(a3 + 24) += 3LL;
        }
      }
      else if ( v6 <= 2 )
      {
        LOBYTE(v6) = sub_16E7EE0(a3, "row", 3u);
      }
      else
      {
        *(_BYTE *)(v11 + 2) = 119;
        *(_WORD *)v11 = 28530;
        *(_QWORD *)(a3 + 24) += 3LL;
      }
    }
    else if ( !strcmp((const char *)a4, "mmarowcol") )
    {
      if ( (_DWORD)v7 == 2 )
      {
        v19 = *(_QWORD *)(a3 + 24);
        v6 = *(_QWORD *)(a3 + 16) - v19;
        if ( v6 <= 6 )
        {
          LOBYTE(v6) = sub_16E7EE0(a3, "col.row", 7u);
        }
        else
        {
          *(_DWORD *)v19 = 778858339;
          *(_WORD *)(v19 + 4) = 28530;
          *(_BYTE *)(v19 + 6) = 119;
          *(_QWORD *)(a3 + 24) += 7LL;
        }
      }
      else if ( (int)v7 > 2 )
      {
        if ( (_DWORD)v7 == 3 )
        {
          v13 = *(_QWORD *)(a3 + 24);
          v6 = *(_QWORD *)(a3 + 16) - v13;
          if ( v6 <= 6 )
          {
            LOBYTE(v6) = sub_16E7EE0(a3, "col.col", 7u);
          }
          else
          {
            *(_DWORD *)v13 = 778858339;
            *(_WORD *)(v13 + 4) = 28515;
            *(_BYTE *)(v13 + 6) = 108;
            *(_QWORD *)(a3 + 24) += 7LL;
          }
        }
      }
      else if ( (_DWORD)v7 )
      {
        if ( (_DWORD)v7 == 1 )
        {
          v12 = *(_QWORD *)(a3 + 24);
          v6 = *(_QWORD *)(a3 + 16) - v12;
          if ( v6 <= 6 )
          {
            LOBYTE(v6) = sub_16E7EE0(a3, "row.col", 7u);
          }
          else
          {
            *(_DWORD *)v12 = 779579250;
            *(_WORD *)(v12 + 4) = 28515;
            *(_BYTE *)(v12 + 6) = 108;
            *(_QWORD *)(a3 + 24) += 7LL;
          }
        }
      }
      else
      {
        v18 = *(_QWORD *)(a3 + 24);
        v6 = *(_QWORD *)(a3 + 16) - v18;
        if ( v6 <= 6 )
        {
          LOBYTE(v6) = sub_16E7EE0(a3, "row.row", 7u);
        }
        else
        {
          *(_DWORD *)v18 = 779579250;
          *(_WORD *)(v18 + 4) = 28530;
          *(_BYTE *)(v18 + 6) = 119;
          *(_QWORD *)(a3 + 24) += 7LL;
        }
      }
    }
    else if ( !strcmp((const char *)a4, "satf") )
    {
      if ( (_DWORD)v7 )
      {
        v16 = *(void **)(a3 + 24);
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v16 <= 9u )
        {
          LOBYTE(v6) = sub_16E7EE0(a3, ".satfinite", 0xAu);
        }
        else
        {
          qmemcpy(v16, ".satfinite", 10);
          *(_QWORD *)(a3 + 24) += 10LL;
          LOBYTE(v6) = 116;
        }
      }
    }
    else if ( !strcmp((const char *)a4, "abtype") )
    {
      v17 = *(char **)(a3 + 24);
      switch ( (int)v7 )
      {
        case 0:
          v22 = *(char **)(a3 + 16);
          v21 = v22 == v17;
          v6 = v22 - v17;
          if ( v21 || v6 == 1 )
          {
            LOBYTE(v6) = sub_16E7EE0(a3, "u8", 2u);
          }
          else
          {
            *(_WORD *)v17 = 14453;
            *(_QWORD *)(a3 + 24) += 2LL;
          }
          break;
        case 1:
          v25 = *(char **)(a3 + 16);
          v21 = v25 == v17;
          v6 = v25 - v17;
          if ( v21 || v6 == 1 )
          {
            LOBYTE(v6) = sub_16E7EE0(a3, "s8", 2u);
          }
          else
          {
            *(_WORD *)v17 = 14451;
            *(_QWORD *)(a3 + 24) += 2LL;
          }
          break;
        case 2:
          v23 = *(char **)(a3 + 16);
          v21 = v23 == v17;
          v6 = v23 - v17;
          if ( v21 || v6 == 1 )
          {
            LOBYTE(v6) = sub_16E7EE0(a3, "u4", 2u);
          }
          else
          {
            *(_WORD *)v17 = 13429;
            *(_QWORD *)(a3 + 24) += 2LL;
          }
          break;
        case 3:
          v24 = *(char **)(a3 + 16);
          v21 = v24 == v17;
          v6 = v24 - v17;
          if ( v21 || v6 == 1 )
          {
            LOBYTE(v6) = sub_16E7EE0(a3, "s4", 2u);
          }
          else
          {
            *(_WORD *)v17 = 13427;
            *(_QWORD *)(a3 + 24) += 2LL;
          }
          break;
        case 4:
          v20 = *(char **)(a3 + 16);
          v21 = v20 == v17;
          v6 = v20 - v17;
          if ( v21 || v6 == 1 )
          {
            LOBYTE(v6) = sub_16E7EE0(a3, "b1", 2u);
          }
          else
          {
            *(_WORD *)v17 = 12642;
            *(_QWORD *)(a3 + 24) += 2LL;
          }
          break;
        case 5:
          v6 = *(_QWORD *)(a3 + 16) - (_QWORD)v17;
          if ( v6 <= 3 )
          {
            LOBYTE(v6) = sub_16E7EE0(a3, "bf16", 4u);
          }
          else
          {
            *(_DWORD *)v17 = 909207138;
            *(_QWORD *)(a3 + 24) += 4LL;
          }
          break;
        case 6:
          v6 = *(_QWORD *)(a3 + 16) - (_QWORD)v17;
          if ( v6 <= 3 )
          {
            LOBYTE(v6) = sub_16E7EE0(a3, "tf32", 4u);
          }
          else
          {
            *(_DWORD *)v17 = 842229364;
            *(_QWORD *)(a3 + 24) += 4LL;
          }
          break;
        default:
          ++*(_DWORD *)(v4 + 72);
          BUG();
      }
    }
    else if ( !strcmp((const char *)a4, "trans") )
    {
      if ( (_DWORD)v7 == 1 )
      {
        v6 = *(_QWORD *)(a3 + 24);
        if ( *(_QWORD *)(a3 + 16) - v6 <= 5 )
        {
          LOBYTE(v6) = sub_16E7EE0(a3, ".trans", 6u);
        }
        else
        {
          *(_DWORD *)v6 = 1634890798;
          *(_WORD *)(v6 + 4) = 29550;
          *(_QWORD *)(a3 + 24) += 6LL;
        }
      }
    }
    else
    {
      v9 = strcmp((const char *)a4, "opcode") == 0;
      LOBYTE(v6) = !v9;
      if ( v9 )
      {
        if ( (_DWORD)v7 == 1 )
        {
          v6 = *(_QWORD *)(a3 + 24);
          if ( *(_QWORD *)(a3 + 16) - v6 <= 2 )
          {
            LOBYTE(v6) = sub_16E7EE0(a3, "and", 3u);
          }
          else
          {
            *(_WORD *)v6 = 28257;
            *(_BYTE *)(v6 + 2) = 100;
            *(_QWORD *)(a3 + 24) += 3LL;
          }
        }
        else if ( (_DWORD)v7 == 2 )
        {
          v6 = *(_QWORD *)(a3 + 24);
          if ( *(_QWORD *)(a3 + 16) - v6 <= 2 )
          {
            LOBYTE(v6) = sub_16E7EE0(a3, "xor", 3u);
          }
          else
          {
            *(_WORD *)v6 = 28536;
            *(_BYTE *)(v6 + 2) = 114;
            *(_QWORD *)(a3 + 24) += 3LL;
          }
        }
      }
    }
  }
  return v6;
}
