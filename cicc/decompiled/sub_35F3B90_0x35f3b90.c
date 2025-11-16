// Function: sub_35F3B90
// Address: 0x35f3b90
//
void __fastcall sub_35F3B90(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  size_t v8; // rax
  int v9; // r13d
  __int64 v10; // rdx
  __int64 v11; // rdx
  size_t v12; // rdx
  char *v13; // rsi
  _DWORD *v14; // rdx
  _DWORD *v15; // rdx

  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  if ( a5 )
  {
    v8 = strlen((const char *)a5);
    if ( v8 == 2 )
    {
      if ( *(_WORD *)a5 != 28783 )
        return;
      v9 = v5 & 7;
      if ( v9 )
      {
        if ( (_BYTE)v9 != 1 )
          BUG();
        v11 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v11) > 2 )
        {
          *(_BYTE *)(v11 + 2) = 120;
          *(_WORD *)v11 = 24941;
          *(_QWORD *)(a4 + 32) += 3LL;
          return;
        }
        v12 = 3;
        v13 = "max";
      }
      else
      {
        v10 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v10) > 2 )
        {
          *(_BYTE *)(v10 + 2) = 110;
          *(_WORD *)v10 = 26989;
          *(_QWORD *)(a4 + 32) += 3LL;
          return;
        }
        v12 = 3;
        v13 = "min";
      }
LABEL_12:
      sub_CB6200(a4, (unsigned __int8 *)v13, v12);
      return;
    }
    if ( v8 == 3 )
    {
      if ( *(_WORD *)a5 == 25185 && *(_BYTE *)(a5 + 2) == 115 )
      {
        if ( (v5 & 8) != 0 )
        {
          v15 = *(_DWORD **)(a4 + 32);
          if ( *(_QWORD *)(a4 + 24) - (_QWORD)v15 > 3u )
          {
            *v15 = 1935827246;
            *(_QWORD *)(a4 + 32) += 4LL;
            return;
          }
          v12 = 4;
          v13 = ".abs";
          goto LABEL_12;
        }
      }
      else if ( *(_WORD *)a5 == 24942 && *(_BYTE *)(a5 + 2) == 110 && (v5 & 0x10) != 0 )
      {
        v14 = *(_DWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v14 > 3u )
        {
          *v14 = 1851878958;
          *(_QWORD *)(a4 + 32) += 4LL;
          return;
        }
        v12 = 4;
        v13 = ".nan";
        goto LABEL_12;
      }
    }
  }
}
