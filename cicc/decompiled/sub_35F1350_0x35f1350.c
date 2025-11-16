// Function: sub_35F1350
// Address: 0x35f1350
//
char __fastcall sub_35F1350(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, const char *a5)
{
  __int64 v6; // rbx
  char v7; // al
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  bool v12; // zf
  _DWORD *v13; // rcx
  unsigned __int64 v14; // rdx
  _DWORD *v15; // rdx

  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  if ( !strcmp(a5, "dim") )
  {
    v7 = v6 & 0xF;
    if ( (v6 & 0xF) == 4 )
    {
      v15 = *(_DWORD **)(a4 + 32);
      v9 = *(_QWORD *)(a4 + 24) - (_QWORD)v15;
      if ( v9 <= 3 )
      {
        LOBYTE(v9) = sub_CB6200(a4, ".a2d", 4u);
      }
      else
      {
        *v15 = 1681023278;
        *(_QWORD *)(a4 + 32) += 4LL;
      }
    }
    else if ( v7 == 5 )
    {
      v8 = *(_QWORD *)(a4 + 32);
      v9 = *(_QWORD *)(a4 + 24) - v8;
      if ( v9 <= 2 )
      {
        LOBYTE(v9) = sub_CB6200(a4, ".3d", 3u);
      }
      else
      {
        *(_BYTE *)(v8 + 2) = 100;
        *(_WORD *)v8 = 13102;
        *(_QWORD *)(a4 + 32) += 3LL;
      }
    }
    else
    {
      if ( v7 != 3 )
        BUG();
      v11 = *(_QWORD *)(a4 + 32);
      v9 = *(_QWORD *)(a4 + 24) - v11;
      if ( v9 <= 2 )
      {
        LOBYTE(v9) = sub_CB6200(a4, ".2d", 3u);
      }
      else
      {
        *(_BYTE *)(v11 + 2) = 100;
        *(_WORD *)v11 = 12846;
        *(_QWORD *)(a4 + 32) += 3LL;
      }
    }
  }
  else if ( !strcmp(a5, "level") )
  {
    LOBYTE(v9) = v6 & 0x30;
    if ( (v6 & 0x30) == 0x20 )
    {
      v10 = *(_QWORD *)(a4 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v10) <= 5 )
      {
        LOBYTE(v9) = sub_CB6200(a4, ".level", 6u);
      }
      else
      {
        *(_DWORD *)v10 = 1986358318;
        *(_WORD *)(v10 + 4) = 27749;
        *(_QWORD *)(a4 + 32) += 6LL;
        LOBYTE(v9) = 101;
      }
    }
  }
  else
  {
    v12 = strcmp(a5, "destty") == 0;
    LOBYTE(v9) = !v12;
    if ( v12 )
    {
      v13 = *(_DWORD **)(a4 + 32);
      v14 = *(_QWORD *)(a4 + 24) - (_QWORD)v13;
      if ( (v6 & 0x8000) != 0 )
      {
        if ( v14 <= 3 )
        {
          LOBYTE(v9) = sub_CB6200(a4, (unsigned __int8 *)".u32", 4u);
        }
        else
        {
          *v13 = 842233134;
          *(_QWORD *)(a4 + 32) += 4LL;
        }
      }
      else if ( v14 <= 3 )
      {
        LOBYTE(v9) = sub_CB6200(a4, ".s32", 4u);
      }
      else
      {
        *v13 = 842232622;
        *(_QWORD *)(a4 + 32) += 4LL;
      }
    }
  }
  return v9;
}
