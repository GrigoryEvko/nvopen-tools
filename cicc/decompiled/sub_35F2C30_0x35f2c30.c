// Function: sub_35F2C30
// Address: 0x35f2c30
//
void __fastcall sub_35F2C30(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rax
  _BYTE *v9; // rcx
  _BYTE *v10; // rax
  __int64 v11; // rcx
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  void *v17; // rdx
  int v18; // kr00_4
  _WORD *v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax

  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  if ( a5 )
  {
    if ( !strcmp((const char *)a5, "addsp") )
    {
      if ( (_DWORD)v6 == 3 )
      {
        v16 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v16) <= 6 )
        {
          sub_CB6200(a4, (unsigned __int8 *)".shared", 7u);
        }
        else
        {
          *(_DWORD *)v16 = 1634235182;
          *(_WORD *)(v16 + 4) = 25970;
          *(_BYTE *)(v16 + 6) = 100;
          *(_QWORD *)(a4 + 32) += 7LL;
        }
      }
      else
      {
        if ( (int)v6 <= 3 )
        {
          if ( !(_DWORD)v6 )
            return;
          if ( (_DWORD)v6 == 1 )
          {
            v7 = *(_QWORD *)(a4 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v7) <= 6 )
            {
              sub_CB6200(a4, (unsigned __int8 *)".global", 7u);
            }
            else
            {
              *(_DWORD *)v7 = 1869375278;
              *(_WORD *)(v7 + 4) = 24930;
              *(_BYTE *)(v7 + 6) = 108;
              *(_QWORD *)(a4 + 32) += 7LL;
            }
            return;
          }
LABEL_101:
          BUG();
        }
        if ( (_DWORD)v6 != 5 )
          goto LABEL_101;
        v13 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v13) <= 5 )
        {
          sub_CB6200(a4, (unsigned __int8 *)".local", 6u);
        }
        else
        {
          *(_DWORD *)v13 = 1668246574;
          *(_WORD *)(v13 + 4) = 27745;
          *(_QWORD *)(a4 + 32) += 6LL;
        }
      }
    }
    else if ( *(_BYTE *)a5 == 97 && *(_BYTE *)(a5 + 1) == 98 && !*(_BYTE *)(a5 + 2) )
    {
      v9 = *(_BYTE **)(a4 + 24);
      v10 = *(_BYTE **)(a4 + 32);
      if ( (_DWORD)v6 )
      {
        if ( v9 == v10 )
        {
          sub_CB6200(a4, (unsigned __int8 *)"b", 1u);
        }
        else
        {
          *v10 = 98;
          ++*(_QWORD *)(a4 + 32);
        }
      }
      else if ( v9 == v10 )
      {
        sub_CB6200(a4, (unsigned __int8 *)"a", 1u);
      }
      else
      {
        *v10 = 97;
        ++*(_QWORD *)(a4 + 32);
      }
    }
    else if ( !strcmp((const char *)a5, "rowcol") )
    {
      v11 = *(_QWORD *)(a4 + 32);
      v12 = *(_QWORD *)(a4 + 24) - v11;
      if ( (_DWORD)v6 )
      {
        if ( v12 <= 2 )
        {
          sub_CB6200(a4, (unsigned __int8 *)"col", 3u);
        }
        else
        {
          *(_BYTE *)(v11 + 2) = 108;
          *(_WORD *)v11 = 28515;
          *(_QWORD *)(a4 + 32) += 3LL;
        }
      }
      else if ( v12 <= 2 )
      {
        sub_CB6200(a4, (unsigned __int8 *)"row", 3u);
      }
      else
      {
        *(_BYTE *)(v11 + 2) = 119;
        *(_WORD *)v11 = 28530;
        *(_QWORD *)(a4 + 32) += 3LL;
      }
    }
    else if ( !strcmp((const char *)a5, "mmarowcol") )
    {
      if ( (_DWORD)v6 == 2 )
      {
        v21 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v21) <= 6 )
        {
          sub_CB6200(a4, "col.row", 7u);
        }
        else
        {
          *(_DWORD *)v21 = 778858339;
          *(_WORD *)(v21 + 4) = 28530;
          *(_BYTE *)(v21 + 6) = 119;
          *(_QWORD *)(a4 + 32) += 7LL;
        }
      }
      else if ( (int)v6 > 2 )
      {
        if ( (_DWORD)v6 == 3 )
        {
          v15 = *(_QWORD *)(a4 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v15) <= 6 )
          {
            sub_CB6200(a4, "col.col", 7u);
          }
          else
          {
            *(_DWORD *)v15 = 778858339;
            *(_WORD *)(v15 + 4) = 28515;
            *(_BYTE *)(v15 + 6) = 108;
            *(_QWORD *)(a4 + 32) += 7LL;
          }
        }
      }
      else if ( (_DWORD)v6 )
      {
        if ( (_DWORD)v6 == 1 )
        {
          v14 = *(_QWORD *)(a4 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v14) <= 6 )
          {
            sub_CB6200(a4, "row.col", 7u);
          }
          else
          {
            *(_DWORD *)v14 = 779579250;
            *(_WORD *)(v14 + 4) = 28515;
            *(_BYTE *)(v14 + 6) = 108;
            *(_QWORD *)(a4 + 32) += 7LL;
          }
        }
      }
      else
      {
        v20 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v20) <= 6 )
        {
          sub_CB6200(a4, "row.row", 7u);
        }
        else
        {
          *(_DWORD *)v20 = 779579250;
          *(_WORD *)(v20 + 4) = 28530;
          *(_BYTE *)(v20 + 6) = 119;
          *(_QWORD *)(a4 + 32) += 7LL;
        }
      }
    }
    else if ( !strcmp((const char *)a5, "satf") )
    {
      if ( (_DWORD)v6 )
      {
        v17 = *(void **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v17 <= 9u )
        {
          sub_CB6200(a4, ".satfinite", 0xAu);
        }
        else
        {
          qmemcpy(v17, ".satfinite", 10);
          *(_QWORD *)(a4 + 32) += 10LL;
        }
      }
    }
    else
    {
      if ( !strcmp((const char *)a5, "abtype") )
      {
        v18 = v6;
        v19 = *(_WORD **)(a4 + 32);
        switch ( v18 )
        {
          case 0:
            if ( *(_QWORD *)(a4 + 24) - (_QWORD)v19 <= 1u )
            {
              sub_CB6200(a4, (unsigned __int8 *)"u8", 2u);
            }
            else
            {
              *v19 = 14453;
              *(_QWORD *)(a4 + 32) += 2LL;
            }
            return;
          case 1:
            if ( *(_QWORD *)(a4 + 24) - (_QWORD)v19 <= 1u )
            {
              sub_CB6200(a4, (unsigned __int8 *)"s8", 2u);
            }
            else
            {
              *v19 = 14451;
              *(_QWORD *)(a4 + 32) += 2LL;
            }
            return;
          case 2:
            if ( *(_QWORD *)(a4 + 24) - (_QWORD)v19 <= 1u )
            {
              sub_CB6200(a4, (unsigned __int8 *)"u4", 2u);
            }
            else
            {
              *v19 = 13429;
              *(_QWORD *)(a4 + 32) += 2LL;
            }
            return;
          case 3:
            if ( *(_QWORD *)(a4 + 24) - (_QWORD)v19 <= 1u )
            {
              sub_CB6200(a4, (unsigned __int8 *)"s4", 2u);
            }
            else
            {
              *v19 = 13427;
              *(_QWORD *)(a4 + 32) += 2LL;
            }
            return;
          case 4:
            if ( *(_QWORD *)(a4 + 24) - (_QWORD)v19 <= 1u )
            {
              sub_CB6200(a4, (unsigned __int8 *)"b1", 2u);
            }
            else
            {
              *v19 = 12642;
              *(_QWORD *)(a4 + 32) += 2LL;
            }
            return;
          case 5:
            if ( *(_QWORD *)(a4 + 24) - (_QWORD)v19 <= 3u )
            {
              sub_CB6200(a4, (unsigned __int8 *)"bf16", 4u);
            }
            else
            {
              *(_DWORD *)v19 = 909207138;
              *(_QWORD *)(a4 + 32) += 4LL;
            }
            return;
          case 6:
            if ( *(_QWORD *)(a4 + 24) - (_QWORD)v19 <= 3u )
            {
              sub_CB6200(a4, (unsigned __int8 *)"tf32", 4u);
            }
            else
            {
              *(_DWORD *)v19 = 842229364;
              *(_QWORD *)(a4 + 32) += 4LL;
            }
            return;
          default:
            goto LABEL_101;
        }
      }
      if ( !strcmp((const char *)a5, "trans") )
      {
        if ( (_DWORD)v6 == 1 )
        {
          v22 = *(_QWORD *)(a4 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v22) <= 5 )
          {
            sub_CB6200(a4, ".trans", 6u);
          }
          else
          {
            *(_DWORD *)v22 = 1634890798;
            *(_WORD *)(v22 + 4) = 29550;
            *(_QWORD *)(a4 + 32) += 6LL;
          }
        }
      }
      else
      {
        if ( strcmp((const char *)a5, "opcode") )
          return;
        switch ( (_DWORD)v6 )
        {
          case 1:
            v23 = *(_QWORD *)(a4 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v23) <= 2 )
            {
              sub_CB6200(a4, (unsigned __int8 *)"and", 3u);
            }
            else
            {
              *(_WORD *)v23 = 28257;
              *(_BYTE *)(v23 + 2) = 100;
              *(_QWORD *)(a4 + 32) += 3LL;
            }
            break;
          case 2:
            v8 = *(_QWORD *)(a4 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v8) <= 2 )
            {
              sub_CB6200(a4, (unsigned __int8 *)"xor", 3u);
            }
            else
            {
              *(_WORD *)v8 = 28536;
              *(_BYTE *)(v8 + 2) = 114;
              *(_QWORD *)(a4 + 32) += 3LL;
            }
            break;
          case 0:
            return;
          default:
            goto LABEL_101;
        }
      }
    }
  }
}
