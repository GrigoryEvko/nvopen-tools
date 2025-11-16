// Function: sub_35F1D30
// Address: 0x35f1d30
//
void __fastcall sub_35F1D30(__int64 a1, __int64 a2, unsigned int a3, _QWORD *a4, __int64 a5)
{
  __int64 v8; // r14
  size_t v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD *v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  size_t v16; // rdx
  char *v17; // rsi
  __int64 v18; // rdx
  _WORD *v19; // rdx
  void *v20; // rdx

  if ( !a5 )
    sub_C64ED0("Empty Modifier", 1u);
  v8 = sub_CE1160(*(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8));
  v9 = strlen((const char *)a5);
  if ( v9 == 2 )
  {
    if ( *(_WORD *)a5 == 29555 )
    {
      v13 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * (a3 + 1) + 8);
      if ( !(_DWORD)v13 )
        return;
      if ( (_DWORD)v13 == 1 )
      {
        v14 = a4[4];
        if ( (unsigned __int64)(a4[3] - v14) > 6 )
        {
          *(_DWORD *)v14 = 1869375278;
          *(_WORD *)(v14 + 4) = 24930;
          *(_BYTE *)(v14 + 6) = 108;
          a4[4] += 7LL;
          return;
        }
        v16 = 7;
        v17 = ".global";
        goto LABEL_23;
      }
    }
    else if ( *(_WORD *)a5 == 25454 )
    {
      if ( (v8 & 0x100) != 0 )
      {
        v15 = a4[4];
        if ( (unsigned __int64)(a4[3] - v15) > 2 )
        {
          *(_BYTE *)(v15 + 2) = 99;
          *(_WORD *)v15 = 28206;
          a4[4] += 3LL;
          return;
        }
        v16 = 3;
        v17 = (char *)&unk_435F0B4;
        goto LABEL_23;
      }
      return;
    }
    goto LABEL_50;
  }
  if ( v9 != 3 )
  {
    if ( v9 == 7 )
    {
      if ( *(_DWORD *)a5 == 1668506980 && *(_WORD *)(a5 + 4) == 30067 && *(_BYTE *)(a5 + 6) == 102 )
      {
        if ( (v8 & 0x400) != 0 )
        {
          v20 = (void *)a4[4];
          if ( a4[3] - (_QWORD)v20 > 0xEu )
          {
            qmemcpy(v20, ".L2::cache_hint", 15);
            a4[4] += 15LL;
            return;
          }
          v16 = 15;
          v17 = ".L2::cache_hint";
LABEL_23:
          sub_CB6200((__int64)a4, (unsigned __int8 *)v17, v16);
          return;
        }
        return;
      }
      if ( *(_DWORD *)a5 == 1718185589 && *(_WORD *)(a5 + 4) == 25961 && *(_BYTE *)(a5 + 6) == 100 )
      {
        if ( (v8 & 0x1000000000LL) != 0 )
        {
          v12 = (_QWORD *)a4[4];
          if ( a4[3] - (_QWORD)v12 > 7u )
          {
            *v12 = 0x64656966696E752ELL;
            a4[4] += 8LL;
            return;
          }
          v16 = 8;
          v17 = ".unified";
          goto LABEL_23;
        }
        return;
      }
    }
    else if ( v9 == 4 && *(_DWORD *)a5 == 1668506980 )
    {
      if ( (v8 & 0x400) != 0 )
      {
        v19 = (_WORD *)a4[4];
        if ( a4[3] - (_QWORD)v19 <= 1u )
        {
          sub_CB6200((__int64)a4, (unsigned __int8 *)", ", 2u);
        }
        else
        {
          *v19 = 8236;
          a4[4] += 2LL;
        }
        sub_35EE840(a1, a2, *(_DWORD *)(a2 + 24) - 1, a4, v10, v11);
      }
      return;
    }
LABEL_50:
    BUG();
  }
  if ( *(_WORD *)a5 != 28534 || *(_BYTE *)(a5 + 2) != 108 )
  {
    if ( *(_WORD *)a5 == 28515 && *(_BYTE *)(a5 + 2) == 112 )
      return;
    goto LABEL_50;
  }
  if ( (v8 & 0x200) != 0 )
  {
    v18 = a4[4];
    if ( (unsigned __int64)(a4[3] - v18) > 8 )
    {
      *(_BYTE *)(v18 + 8) = 101;
      *(_QWORD *)v18 = 0x6C6974616C6F762ELL;
      a4[4] += 9LL;
      return;
    }
    v16 = 9;
    v17 = ".volatile";
    goto LABEL_23;
  }
}
