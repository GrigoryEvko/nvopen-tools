// Function: sub_1A50660
// Address: 0x1a50660
//
char *__fastcall sub_1A50660(char *src, char *a2, char *a3, char *a4, _QWORD *a5, __int64 a6)
{
  char *v6; // r10
  char *v8; // r12
  int v11; // ecx
  __int64 v12; // r9
  __int64 v13; // rsi
  int v14; // ecx
  __int64 v15; // r14
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // r15
  _QWORD **v19; // rax
  _QWORD *v20; // rdx
  unsigned int i; // eax
  unsigned int v22; // edi
  __int64 *v23; // rdx
  __int64 v24; // r15
  _QWORD **v25; // rdx
  _QWORD *v26; // rdx
  unsigned int j; // ecx
  signed __int64 v28; // r13
  char *v29; // r8
  int v31; // edx
  int v32; // eax
  int v33; // edi
  int v34; // [rsp+Ch] [rbp-34h]

  v6 = src;
  v8 = a3;
  if ( a3 != a4 && src != a2 )
  {
    while ( 1 )
    {
      v11 = *(_DWORD *)(a6 + 24);
      v12 = *(_QWORD *)v6;
      if ( !v11 )
        goto LABEL_23;
      v13 = *(_QWORD *)v8;
      v14 = v11 - 1;
      v15 = *(_QWORD *)(a6 + 8);
      v16 = v14 & (((unsigned int)*(_QWORD *)v8 >> 9) ^ ((unsigned int)*(_QWORD *)v8 >> 4));
      v17 = (__int64 *)(v15 + 16LL * v16);
      v18 = *v17;
      if ( *(_QWORD *)v8 != *v17 )
        break;
LABEL_5:
      v19 = (_QWORD **)v17[1];
      if ( !v19 )
        goto LABEL_26;
      v20 = *v19;
      for ( i = 1; v20; ++i )
        v20 = (_QWORD *)*v20;
LABEL_8:
      v22 = v14 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v23 = (__int64 *)(v15 + 16LL * v22);
      v24 = *v23;
      if ( v12 == *v23 )
      {
LABEL_9:
        v25 = (_QWORD **)v23[1];
        if ( v25 )
        {
          v26 = *v25;
          for ( j = 1; v26; ++j )
            v26 = (_QWORD *)*v26;
          if ( j > i )
          {
            v8 += 8;
            goto LABEL_14;
          }
        }
      }
      else
      {
        v31 = 1;
        while ( v24 != -8 )
        {
          v22 = v14 & (v31 + v22);
          v34 = v31 + 1;
          v23 = (__int64 *)(v15 + 16LL * v22);
          v24 = *v23;
          if ( v12 == *v23 )
            goto LABEL_9;
          v31 = v34;
        }
      }
LABEL_23:
      v6 += 8;
      v13 = v12;
LABEL_14:
      *a5++ = v13;
      if ( v6 == a2 || v8 == a4 )
        goto LABEL_16;
    }
    v32 = 1;
    while ( v18 != -8 )
    {
      v33 = v32 + 1;
      v16 = v14 & (v32 + v16);
      v17 = (__int64 *)(v15 + 16LL * v16);
      v18 = *v17;
      if ( v13 == *v17 )
        goto LABEL_5;
      v32 = v33;
    }
LABEL_26:
    i = 0;
    goto LABEL_8;
  }
LABEL_16:
  v28 = a2 - v6;
  if ( a2 != v6 )
    a5 = memmove(a5, v6, a2 - v6);
  v29 = (char *)a5 + v28;
  if ( a4 != v8 )
    v29 = (char *)memmove(v29, v8, a4 - v8);
  return &v29[a4 - v8];
}
