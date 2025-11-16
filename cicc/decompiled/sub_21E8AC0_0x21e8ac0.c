// Function: sub_21E8AC0
// Address: 0x21e8ac0
//
char __fastcall sub_21E8AC0(__int64 a1, unsigned int a2, __int64 a3, const char *a4)
{
  unsigned __int64 v4; // rax
  __int64 v6; // rdx
  char v7; // dl
  bool v8; // zf
  __int64 v9; // rax
  __int64 v10; // rdx
  _DWORD *v11; // rcx
  _DWORD *v12; // rdx
  __int64 v13; // rdx

  LOBYTE(v4) = (_BYTE)a4;
  v6 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL * a2 + 8);
  if ( !strcmp(a4, "dim") )
  {
    v7 = v6 & 0xF;
    if ( v7 == 4 )
    {
      v12 = *(_DWORD **)(a3 + 24);
      v4 = *(_QWORD *)(a3 + 16) - (_QWORD)v12;
      if ( v4 <= 3 )
      {
        LOBYTE(v4) = sub_16E7EE0(a3, ".a2d", 4u);
      }
      else
      {
        *v12 = 1681023278;
        *(_QWORD *)(a3 + 24) += 4LL;
      }
    }
    else
    {
      v8 = v7 == 5;
      v9 = *(_QWORD *)(a3 + 16);
      v10 = *(_QWORD *)(a3 + 24);
      if ( v8 )
      {
        v4 = v9 - v10;
        if ( v4 <= 2 )
        {
          LOBYTE(v4) = sub_16E7EE0(a3, ".3d", 3u);
        }
        else
        {
          *(_BYTE *)(v10 + 2) = 100;
          *(_WORD *)v10 = 13102;
          *(_QWORD *)(a3 + 24) += 3LL;
        }
      }
      else
      {
        v4 = v9 - v10;
        if ( v4 <= 2 )
        {
          LOBYTE(v4) = sub_16E7EE0(a3, ".2d", 3u);
        }
        else
        {
          *(_BYTE *)(v10 + 2) = 100;
          *(_WORD *)v10 = 12846;
          *(_QWORD *)(a3 + 24) += 3LL;
        }
      }
    }
  }
  else if ( !strcmp(a4, "level") )
  {
    if ( (v6 & 0x30) == 0x20 )
    {
      v13 = *(_QWORD *)(a3 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v13) <= 5 )
      {
        LOBYTE(v4) = sub_16E7EE0(a3, ".level", 6u);
      }
      else
      {
        *(_DWORD *)v13 = 1986358318;
        *(_WORD *)(v13 + 4) = 27749;
        *(_QWORD *)(a3 + 24) += 6LL;
        LOBYTE(v4) = 101;
      }
    }
  }
  else
  {
    v8 = strcmp(a4, "destty") == 0;
    LOBYTE(v4) = !v8;
    if ( v8 )
    {
      v11 = *(_DWORD **)(a3 + 24);
      v4 = *(_QWORD *)(a3 + 16) - (_QWORD)v11;
      if ( (v6 & 0x8000) != 0 )
      {
        if ( v4 <= 3 )
        {
          LOBYTE(v4) = sub_16E7EE0(a3, ".u32", 4u);
        }
        else
        {
          *v11 = 842233134;
          *(_QWORD *)(a3 + 24) += 4LL;
        }
      }
      else if ( v4 <= 3 )
      {
        LOBYTE(v4) = sub_16E7EE0(a3, ".s32", 4u);
      }
      else
      {
        *v11 = 842232622;
        *(_QWORD *)(a3 + 24) += 4LL;
      }
    }
  }
  return v4;
}
