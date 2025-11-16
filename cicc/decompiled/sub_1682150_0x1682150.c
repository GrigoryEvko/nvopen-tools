// Function: sub_1682150
// Address: 0x1682150
//
__int64 __fastcall sub_1682150(char *src)
{
  const char *v1; // r14
  __int64 v2; // r13
  int v3; // r12d
  __int64 v4; // rax
  int v5; // r12d
  char *v6; // rdi
  char *v7; // rax
  char *v8; // rax
  char *v9; // rbx
  int v10; // ebx
  __int64 v11; // rax
  char *v12; // rdi
  char *v14; // rax

  if ( !src )
    return 0;
  v1 = src;
  v2 = sub_1687490(sub_1688100, sub_16881F0, 16);
  while ( 1 )
  {
    v8 = strchr(v1, 44);
    v9 = v8;
    if ( !v8 )
      break;
    v3 = (int)v8;
    v4 = sub_1689050();
    v5 = v3 - (_DWORD)v1;
    v6 = (char *)sub_1685080(*(_QWORD *)(v4 + 24), v5 + 1);
    if ( v6 )
    {
      v7 = strncpy(v6, v1, v5);
      v7[v5] = 0;
      sub_1687E30(v7, v2);
    }
    else
    {
      sub_1683C30();
      strncpy(0, v1, v5);
      *(_BYTE *)v5 = 0;
      sub_1687E30(0, v2);
    }
    v1 = v9 + 1;
  }
  v10 = strlen(v1);
  v11 = sub_1689050();
  v12 = (char *)sub_1685080(*(_QWORD *)(v11 + 24), v10 + 1);
  if ( v12 )
  {
    v14 = strncpy(v12, v1, v10);
    v14[v10] = 0;
    sub_1687E30(v14, v2);
  }
  else
  {
    sub_1683C30();
    strncpy(0, v1, v10);
    *(_BYTE *)v10 = 0;
    sub_1687E30(0, v2);
  }
  return v2;
}
