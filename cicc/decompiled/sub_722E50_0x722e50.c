// Function: sub_722E50
// Address: 0x722e50
//
_BOOL8 __fastcall sub_722E50(char *s, char *a2, int a3, int a4, int a5)
{
  char *v7; // r15
  char v8; // al
  char *v9; // r8
  char v10; // al
  char *v11; // rax
  int v12; // eax
  char *v13; // r8
  _BOOL8 result; // rax
  char *v15; // r15
  char *v16; // rax
  unsigned __int8 *v17; // r15
  unsigned __int8 *v18; // rax
  unsigned __int8 *v19; // rbx
  unsigned __int8 *v20; // rax
  int v21; // eax
  char *v22; // [rsp+8h] [rbp-48h]
  const char *s1; // [rsp+10h] [rbp-40h]
  char v24; // [rsp+1Eh] [rbp-32h]
  char v25; // [rsp+1Fh] [rbp-31h]

  if ( a3 )
  {
    v7 = &s[strlen(s) - 1];
    v8 = *v7;
    *v7 = 0;
    v25 = v8;
    v9 = &a2[strlen(a2) - 1];
    v10 = *v9;
    *v9 = 0;
    v22 = v9;
    v24 = v10;
    s1 = sub_722280(s + 1);
    v11 = sub_722280(a2 + 1);
    v12 = strcmp(s1, v11);
    v13 = v22;
    if ( v12 )
    {
      result = 1;
    }
    else
    {
      v19 = (unsigned __int8 *)sub_722430(s + 1, a5);
      v20 = (unsigned __int8 *)sub_722430(a2 + 1, a5);
      v21 = sub_722B80(v19, v20, a4);
      v13 = v22;
      result = v21 != 0;
    }
    *v7 = v25;
    *v13 = v24;
  }
  else
  {
    v15 = sub_722280(s);
    v16 = sub_722280(a2);
    if ( !strcmp(v15, v16) )
    {
      v17 = (unsigned __int8 *)sub_722430(s, a5);
      v18 = (unsigned __int8 *)sub_722430(a2, a5);
      return sub_722B80(v17, v18, a4) != 0;
    }
    else
    {
      return 1;
    }
  }
  return result;
}
