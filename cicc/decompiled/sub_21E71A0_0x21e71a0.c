// Function: sub_21E71A0
// Address: 0x21e71a0
//
char __fastcall sub_21E71A0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, const char *a5)
{
  __int64 v5; // r14
  __int64 v7; // r9
  signed __int64 v8; // rax
  char v9; // al
  _WORD *v10; // rdx
  char *v11; // rdx
  char *v12; // rax
  bool v13; // cf
  _WORD *v14; // rdx
  bool v15; // zf
  char *v16; // rdx
  __int64 v17; // rax
  char *v18; // rax
  char *v19; // rsi
  char *v20; // rax

  v5 = a3;
  v7 = *(_QWORD *)(a2 + 16);
  v8 = *(_QWORD *)(v7 + 88);
  if ( !strcmp(a5, "coords3d") )
  {
    LOBYTE(v8) = v8 & 0xF;
    if ( (_BYTE)v8 != 5 )
      return v8;
  }
  else
  {
    if ( strcmp(a5, "coords2d") )
    {
      if ( !strcmp(a5, "arrayidx") )
      {
        LOBYTE(v8) = v8 & 0xF;
        if ( (_BYTE)v8 != 4 )
          return v8;
        sub_21897A0(a1, a2, a3, a4);
        v11 = *(char **)(a4 + 24);
        v12 = *(char **)(a4 + 16);
        v13 = v12 == v11;
        v8 = v12 - v11;
        if ( !v13 && v8 != 1 )
        {
          *(_WORD *)v11 = 8236;
          *(_QWORD *)(a4 + 24) += 2LL;
          return v8;
        }
        v19 = ", ";
      }
      else
      {
        if ( !strcmp(a5, "lod") )
        {
          LOBYTE(v8) = v8 & 0x30;
          if ( (_BYTE)v8 != 32 )
            return v8;
          goto LABEL_7;
        }
        v15 = strcmp(a5, "component") == 0;
        LOBYTE(v8) = !v15;
        if ( !v15 )
          return v8;
        v16 = *(char **)(a4 + 24);
        v17 = *(_QWORD *)(v7 + 16 * v5 + 8);
        if ( v17 == 2 )
        {
          v20 = *(char **)(a4 + 16);
          v13 = v20 == v16;
          v8 = v20 - v16;
          if ( !v13 && v8 != 1 )
          {
            *(_WORD *)v16 = 25134;
            *(_QWORD *)(a4 + 24) += 2LL;
            return v8;
          }
          v19 = ".b";
        }
        else if ( v17 > 2 )
        {
          if ( *(_QWORD *)(a4 + 16) - (_QWORD)v16 > 1u )
          {
            *(_WORD *)v16 = 24878;
            *(_QWORD *)(a4 + 24) += 2LL;
            LOBYTE(v8) = 46;
            return v8;
          }
          v19 = ".a";
        }
        else
        {
          v15 = v17 == 0;
          v18 = *(char **)(a4 + 16);
          if ( v15 )
          {
            v13 = v18 == v16;
            v8 = v18 - v16;
            if ( !v13 && v8 != 1 )
            {
              *(_WORD *)v16 = 29230;
              *(_QWORD *)(a4 + 24) += 2LL;
              return v8;
            }
            v19 = ".r";
          }
          else
          {
            v13 = v18 == v16;
            v8 = v18 - v16;
            if ( !v13 && v8 != 1 )
            {
              *(_WORD *)v16 = 26414;
              *(_QWORD *)(a4 + 24) += 2LL;
              return v8;
            }
            v19 = ".g";
          }
        }
      }
      LOBYTE(v8) = sub_16E7EE0(a4, v19, 2u);
      return v8;
    }
    v9 = v8 & 0xF;
    if ( v9 != 4 )
    {
      LOBYTE(v8) = v9 - 3;
      if ( (v8 & 0xFD) != 0 )
        return v8;
      goto LABEL_7;
    }
  }
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
  sub_21897A0(a1, a2, v5, a4);
LABEL_7:
  v10 = *(_WORD **)(a4 + 24);
  if ( *(_QWORD *)(a4 + 16) - (_QWORD)v10 <= 1u )
  {
    sub_16E7EE0(a4, ", ", 2u);
  }
  else
  {
    *v10 = 8236;
    *(_QWORD *)(a4 + 24) += 2LL;
  }
  LOBYTE(v8) = (unsigned __int8)sub_21897A0(a1, a2, v5, a4);
  return v8;
}
