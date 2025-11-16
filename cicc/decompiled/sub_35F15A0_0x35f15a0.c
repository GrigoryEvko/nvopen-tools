// Function: sub_35F15A0
// Address: 0x35f15a0
//
void __fastcall sub_35F15A0(__int64 a1, __int64 a2, unsigned int a3, _QWORD *a4, const char *a5)
{
  __int64 v7; // r9
  __int64 v8; // rax
  char v9; // al
  _WORD *v10; // rdx
  __int64 v11; // r8
  _WORD *v12; // rdx
  _WORD *v13; // rdx
  __int64 v14; // rax
  _WORD *v15; // rdx
  _WORD *v16; // rdx
  _WORD *v17; // rdx
  char *v18; // rsi
  _WORD *v19; // rdx

  v7 = *(_QWORD *)(a2 + 16);
  v8 = *(_QWORD *)(v7 + 88);
  if ( !strcmp(a5, "coords3d") )
  {
    if ( (v8 & 0xF) != 5 )
      return;
    goto LABEL_17;
  }
  if ( strcmp(a5, "coords2d") )
  {
    if ( !strcmp(a5, "arrayidx") )
    {
      if ( (v8 & 0xF) != 4 )
        return;
      sub_35EE840(a1, a2, a3, a4, (__int64)a5, v7);
      v12 = (_WORD *)a4[4];
      if ( a4[3] - (_QWORD)v12 > 1u )
      {
        *v12 = 8236;
        a4[4] += 2LL;
        return;
      }
      v18 = ", ";
    }
    else
    {
      if ( !strcmp(a5, "lod") )
      {
        if ( (v8 & 0x30) != 0x20 )
          return;
        goto LABEL_7;
      }
      if ( strcmp(a5, "component") )
        return;
      v14 = *(_QWORD *)(v7 + 16LL * a3 + 8);
      if ( v14 != 2 )
      {
        if ( v14 > 2 )
        {
          if ( v14 == 3 )
          {
            v17 = (_WORD *)a4[4];
            if ( a4[3] - (_QWORD)v17 > 1u )
            {
              *v17 = 24878;
              a4[4] += 2LL;
              return;
            }
            v18 = ".a";
            goto LABEL_35;
          }
        }
        else
        {
          if ( !v14 )
          {
            v15 = (_WORD *)a4[4];
            if ( a4[3] - (_QWORD)v15 > 1u )
            {
              *v15 = 29230;
              a4[4] += 2LL;
              return;
            }
            v18 = ".r";
            goto LABEL_35;
          }
          if ( v14 == 1 )
          {
            v16 = (_WORD *)a4[4];
            if ( a4[3] - (_QWORD)v16 > 1u )
            {
              *v16 = 26414;
              a4[4] += 2LL;
              return;
            }
            v18 = ".g";
            goto LABEL_35;
          }
        }
        BUG();
      }
      v19 = (_WORD *)a4[4];
      if ( a4[3] - (_QWORD)v19 > 1u )
      {
        *v19 = 25134;
        a4[4] += 2LL;
        return;
      }
      v18 = ".b";
    }
LABEL_35:
    sub_CB6200((__int64)a4, (unsigned __int8 *)v18, 2u);
    return;
  }
  v9 = v8 & 0xF;
  if ( v9 != 4 )
  {
    if ( ((v9 - 3) & 0xFD) != 0 )
      return;
    goto LABEL_7;
  }
LABEL_17:
  v13 = (_WORD *)a4[4];
  if ( a4[3] - (_QWORD)v13 <= 1u )
  {
    sub_CB6200((__int64)a4, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v13 = 8236;
    a4[4] += 2LL;
  }
  sub_35EE840(a1, a2, a3, a4, (__int64)a5, v7);
LABEL_7:
  v10 = (_WORD *)a4[4];
  if ( a4[3] - (_QWORD)v10 <= 1u )
  {
    sub_CB6200((__int64)a4, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    v11 = 8236;
    *v10 = 8236;
    a4[4] += 2LL;
  }
  sub_35EE840(a1, a2, a3, a4, v11, v7);
}
