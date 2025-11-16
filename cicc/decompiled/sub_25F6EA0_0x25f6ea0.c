// Function: sub_25F6EA0
// Address: 0x25f6ea0
//
char *__fastcall sub_25F6EA0(char *src, char *a2, char *a3, char *a4, _QWORD *a5)
{
  char *v5; // r15
  unsigned __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r13
  unsigned __int64 v11; // rcx
  int v12; // eax
  unsigned int v13; // edx
  char *v14; // r12
  __int64 v15; // rbx
  int v16; // eax
  void *v17; // rdi
  unsigned __int64 v19; // [rsp+0h] [rbp-50h]
  unsigned int v21; // [rsp+1Ch] [rbp-34h]
  unsigned int v22; // [rsp+1Ch] [rbp-34h]

  v5 = src;
  if ( a2 != src )
  {
    while ( 1 )
    {
      if ( a4 == a3 )
      {
        v17 = a5;
        v14 = (char *)a5 + a2 - v5;
        v15 = 0;
        memmove(v17, v5, a2 - v5);
        return &v14[v15];
      }
      v9 = *(_QWORD *)a3;
      v10 = *(_QWORD *)v5;
      v21 = *(_DWORD *)(*(_QWORD *)a3 + 32LL);
      if ( v21 > 0x40 )
      {
        v16 = sub_C444A0(v9 + 24);
        v11 = -1;
        if ( v21 - v16 <= 0x40 )
          v11 = **(_QWORD **)(v9 + 24);
      }
      else
      {
        v11 = *(_QWORD *)(v9 + 24);
      }
      v22 = *(_DWORD *)(v10 + 32);
      if ( v22 <= 0x40 )
        break;
      v19 = v11;
      v12 = sub_C444A0(v10 + 24);
      v11 = v19;
      v13 = v22 - v12;
      v8 = -1;
      if ( v13 > 0x40 )
      {
LABEL_4:
        if ( v8 > v11 )
          goto LABEL_5;
LABEL_12:
        v5 += 8;
        *a5++ = v10;
        if ( a2 == v5 )
          goto LABEL_13;
      }
      else
      {
        if ( **(_QWORD **)(v10 + 24) <= v19 )
          goto LABEL_12;
LABEL_5:
        *a5 = v9;
        a3 += 8;
        ++a5;
        if ( a2 == v5 )
          goto LABEL_13;
      }
    }
    v8 = *(_QWORD *)(v10 + 24);
    goto LABEL_4;
  }
LABEL_13:
  v14 = (char *)a5 + a2 - v5;
  v15 = a4 - a3;
  if ( a4 != a3 )
    memmove(v14, a3, a4 - a3);
  return &v14[v15];
}
