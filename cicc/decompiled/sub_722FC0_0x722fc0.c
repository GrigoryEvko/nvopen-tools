// Function: sub_722FC0
// Address: 0x722fc0
//
void __fastcall sub_722FC0(char *a1, _QWORD *a2, int a3, int a4)
{
  char v4; // r14
  char *v5; // rbx
  size_t v7; // rax
  char *v8; // rax
  int v9; // r14d
  __int64 v10; // rax
  char *v11; // r15
  char v12; // si
  char *v13; // rax
  __int64 v14; // rax
  int v17; // [rsp+1Ch] [rbp-54h] BYREF
  char s[80]; // [rsp+20h] [rbp-50h] BYREF

  v4 = *a1;
  if ( *a1 )
  {
    v5 = a1;
    while ( 1 )
    {
      if ( a4 && !isprint((unsigned __int8)v4) )
      {
        if ( v4 == 10 )
        {
          sub_8238B0(a2, "\\n", 2);
        }
        else
        {
          sprintf(s, "\\%03o", ((1 << unk_4F06B9C) - 1) & v4);
          v7 = strlen(s);
          sub_8238B0(a2, s, v7);
        }
        v8 = v5++;
        goto LABEL_7;
      }
      if ( a3 && (v4 == 34 || v4 == 92) )
      {
        v14 = a2[2];
        if ( (unsigned __int64)(v14 + 1) > a2[1] )
        {
          sub_823810(a2);
          v14 = a2[2];
        }
        *(_BYTE *)(a2[4] + v14) = 92;
        ++a2[2];
      }
      if ( *v5 >= 0 )
        break;
      v9 = sub_721AB0(v5, &v17, 0);
      v8 = v5 - 1;
      if ( v9 > 0 )
      {
LABEL_14:
        v10 = a2[2];
        v11 = v5;
        do
        {
          if ( (unsigned __int64)(v10 + 1) > a2[1] )
          {
            sub_823810(a2);
            v10 = a2[2];
          }
          v12 = *v11++;
          *(_BYTE *)(a2[4] + v10) = v12;
          v10 = a2[2] + 1LL;
          a2[2] = v10;
        }
        while ( v9 + (int)v5 - (int)v11 > 0 );
        v13 = &v5[v9 - 1];
        v5 += v9;
        v4 = v13[1];
        if ( !v4 )
          return;
      }
      else
      {
LABEL_7:
        v4 = v8[1];
        if ( !v4 )
          return;
      }
    }
    v17 = 0;
    v9 = 1;
    goto LABEL_14;
  }
}
