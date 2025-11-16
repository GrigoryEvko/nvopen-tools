// Function: sub_1C13CF0
// Address: 0x1c13cf0
//
void __fastcall sub_1C13CF0(char **a1, const void **a2)
{
  _BYTE *v4; // rdx
  const void *v5; // r14
  char *v6; // rdi
  signed __int64 v7; // r13
  unsigned __int64 v8; // rsi
  char *v9; // r9
  char *v10; // r13
  char *v11; // r12
  char *v12; // r13
  char *v13; // rsi
  char *v14; // rsi
  size_t v15; // rdx

  if ( a2 != (const void **)a1 )
  {
    v4 = a2[1];
    v5 = *a2;
    v6 = *a1;
    v7 = v4 - (_BYTE *)v5;
    v8 = a1[2] - v6;
    if ( v4 - (_BYTE *)v5 <= v8 )
    {
      v9 = a1[1];
      if ( v7 > (unsigned __int64)(v9 - v6) )
      {
        v13 = 0;
        if ( v9 != v6 )
        {
          memmove(v6, v5, a1[1] - v6);
          v9 = a1[1];
          v6 = *a1;
          v4 = a2[1];
          v5 = *a2;
          v13 = (char *)(v9 - *a1);
        }
        v14 = &v13[(_QWORD)v5];
        v15 = v4 - v14;
        if ( v15 )
        {
          memmove(v9, v14, v15);
          v10 = &(*a1)[v7];
          goto LABEL_7;
        }
      }
      else if ( v7 )
      {
        memmove(v6, v5, v4 - (_BYTE *)v5);
        v6 = *a1;
      }
      v10 = &v6[v7];
LABEL_7:
      a1[1] = v10;
      return;
    }
    if ( v7 )
    {
      if ( v7 < 0 )
        sub_4261EA(v6, v8, v4);
      v11 = (char *)sub_22077B0(v4 - (_BYTE *)v5);
      memcpy(v11, v5, v7);
      v6 = *a1;
      v8 = a1[2] - *a1;
    }
    else
    {
      v11 = 0;
    }
    if ( v6 )
      j_j___libc_free_0(v6, v8);
    v12 = &v11[v7];
    *a1 = v11;
    a1[2] = v12;
    a1[1] = v12;
  }
}
