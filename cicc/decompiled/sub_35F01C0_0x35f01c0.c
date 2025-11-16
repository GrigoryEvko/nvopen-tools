// Function: sub_35F01C0
// Address: 0x35f01c0
//
void __fastcall sub_35F01C0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, const char *a5)
{
  char v8; // al
  __int64 v9; // rdx
  _DWORD *v10; // rdx
  size_t v11; // rdx
  char *v12; // rsi
  __int64 v13; // rdx

  if ( a5 && strlen(a5) == 4 && *(_DWORD *)a5 == 1684957547 )
  {
    v8 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8) & 0xF;
    if ( v8 == 1 )
    {
      v13 = *(_QWORD *)(a4 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v13) > 5 )
      {
        *(_DWORD *)v13 = 1769108065;
        *(_WORD *)(v13 + 4) = 25974;
        *(_QWORD *)(a4 + 32) += 6LL;
        return;
      }
      v11 = 6;
      v12 = "arrive";
    }
    else if ( v8 == 2 )
    {
      v9 = *(_QWORD *)(a4 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v9) > 2 )
      {
        *(_BYTE *)(v9 + 2) = 100;
        *(_WORD *)v9 = 25970;
        *(_QWORD *)(a4 + 32) += 3LL;
        return;
      }
      v11 = 3;
      v12 = "red";
    }
    else
    {
      if ( v8 )
        BUG();
      v10 = *(_DWORD **)(a4 + 32);
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v10 > 3u )
      {
        *v10 = 1668184435;
        *(_QWORD *)(a4 + 32) += 4LL;
        return;
      }
      v11 = 4;
      v12 = "sync";
    }
    sub_CB6200(a4, (unsigned __int8 *)v12, v11);
  }
}
