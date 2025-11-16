// Function: sub_8EE130
// Address: 0x8ee130
//
char *__fastcall sub_8EE130(unsigned __int8 *a1, char *a2, __int64 *a3, int *a4)
{
  char *v4; // r12
  __int64 v5; // r14
  __int64 v6; // r15
  char *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r14
  int v12; // ebx
  char *v14; // rdi
  size_t v15; // rax
  unsigned int v16; // r13d
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  char *v21; // rax
  __int64 v22; // rcx
  bool v25; // [rsp+1Fh] [rbp-151h]
  _BOOL4 v27; // [rsp+30h] [rbp-140h] BYREF
  unsigned int v28; // [rsp+34h] [rbp-13Ch] BYREF
  char *v29; // [rsp+38h] [rbp-138h] BYREF
  _DWORD v30[76]; // [rsp+40h] [rbp-130h] BYREF

  v25 = a2 != 0;
  if ( !a3 && a2 )
  {
    v4 = 0;
    v12 = -3;
  }
  else
  {
    v4 = a2;
    if ( a2 )
    {
      v5 = *a3;
    }
    else
    {
      v5 = 256;
      v4 = (char *)v30;
    }
    v6 = v5;
    while ( 1 )
    {
      v7 = v4;
      sub_8EDED0(a1, v4, v6, &v27, &v28, &v29);
      if ( !v28 )
        break;
      v6 = (__int64)v29;
      if ( v4 == a2 || v4 == (char *)v30 )
      {
        v4 = (char *)malloc(v29, v4, v8, v28, v9, v10);
      }
      else
      {
        v7 = v29;
        v4 = (char *)realloc(v4);
      }
      if ( !v4 )
        goto LABEL_38;
      if ( !v27 )
      {
        v11 = v6;
        goto LABEL_13;
      }
    }
    v11 = v6;
    if ( v27 )
    {
      v12 = -2;
      if ( v4 == (char *)v30 )
      {
LABEL_26:
        v4 = 0;
        goto LABEL_18;
      }
LABEL_24:
      if ( a2 != v4 )
      {
        v14 = v4;
        v4 = 0;
        _libc_free(v14, v7);
        goto LABEL_18;
      }
      goto LABEL_26;
    }
LABEL_13:
    if ( v4 == (char *)v30 )
    {
      v15 = strlen(v4);
      v16 = v15 + 1;
      v21 = (char *)malloc(v15 + 1, v7, v17, v18, v19, v20);
      v4 = v21;
      if ( !v21 )
      {
LABEL_38:
        v4 = 0;
        v12 = -1;
        goto LABEL_24;
      }
      if ( v16 >= 8 )
      {
        *(_QWORD *)&v21[v16 - 8] = *(_QWORD *)((char *)&v30[-2] + v16);
        v22 = (v16 - 1) >> 3;
        qmemcpy(v21, v30, 8 * v22);
        v7 = (char *)&v30[2 * v22];
      }
      else if ( (v16 & 4) != 0 )
      {
        *(_DWORD *)v21 = v30[0];
        *(_DWORD *)&v21[v16 - 4] = *(_DWORD *)((char *)&v30[-1] + v16);
      }
      else if ( v16 )
      {
        *v21 = v30[0];
        if ( (v16 & 2) != 0 )
          *(_WORD *)&v21[v16 - 2] = *(_WORD *)((char *)&v29 + v16 + 6);
      }
    }
    if ( a2 != v4 && v25 )
    {
      v12 = 0;
      _libc_free(a2, v7);
      if ( a3 )
        *a3 = v11;
    }
    else
    {
      v12 = 0;
    }
  }
LABEL_18:
  if ( a4 )
    *a4 = v12;
  return v4;
}
