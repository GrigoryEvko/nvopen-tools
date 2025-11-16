// Function: sub_2957EE0
// Address: 0x2957ee0
//
char *__fastcall sub_2957EE0(char *src, char *a2, char *a3, char *a4, _QWORD *a5, __int64 a6)
{
  char *v6; // r10
  char *v8; // r12
  int v11; // ecx
  __int64 v12; // r9
  __int64 v13; // rsi
  __int64 v14; // r13
  int v15; // ecx
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // r15
  _QWORD **v19; // rax
  _QWORD *v20; // rdx
  unsigned int v21; // edi
  __int64 *v22; // rdx
  __int64 v23; // r15
  _QWORD **v24; // rdx
  _QWORD *v25; // rdx
  unsigned int i; // ecx
  signed __int64 v27; // r13
  char *v28; // r8
  int v30; // edx
  int v31; // eax
  int v32; // edi
  int v33; // [rsp+Ch] [rbp-34h]

  v6 = src;
  v8 = a3;
  if ( a3 != a4 && src != a2 )
  {
    do
    {
      v11 = *(_DWORD *)(a6 + 24);
      v12 = *(_QWORD *)v6;
      v13 = *(_QWORD *)v8;
      v14 = *(_QWORD *)(a6 + 8);
      if ( v11 )
      {
        v15 = v11 - 1;
        v16 = v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v17 = (__int64 *)(v14 + 16LL * v16);
        v18 = *v17;
        if ( v13 == *v17 )
        {
LABEL_5:
          v19 = (_QWORD **)v17[1];
          if ( v19 )
          {
            v20 = *v19;
            for ( LODWORD(v19) = 1; v20; LODWORD(v19) = (_DWORD)v19 + 1 )
              v20 = (_QWORD *)*v20;
          }
        }
        else
        {
          v31 = 1;
          while ( v18 != -4096 )
          {
            v32 = v31 + 1;
            v16 = v15 & (v31 + v16);
            v17 = (__int64 *)(v14 + 16LL * v16);
            v18 = *v17;
            if ( v13 == *v17 )
              goto LABEL_5;
            v31 = v32;
          }
          LODWORD(v19) = 0;
        }
        v21 = v15 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v22 = (__int64 *)(v14 + 16LL * v21);
        v23 = *v22;
        if ( v12 == *v22 )
        {
LABEL_9:
          v24 = (_QWORD **)v22[1];
          if ( v24 )
          {
            v25 = *v24;
            for ( i = 1; v25; ++i )
              v25 = (_QWORD *)*v25;
            if ( i > (unsigned int)v19 )
            {
              v8 += 8;
              goto LABEL_14;
            }
          }
        }
        else
        {
          v30 = 1;
          while ( v23 != -4096 )
          {
            v21 = v15 & (v30 + v21);
            v33 = v30 + 1;
            v22 = (__int64 *)(v14 + 16LL * v21);
            v23 = *v22;
            if ( v12 == *v22 )
              goto LABEL_9;
            v30 = v33;
          }
        }
      }
      v6 += 8;
      v13 = v12;
LABEL_14:
      *a5++ = v13;
    }
    while ( v6 != a2 && v8 != a4 );
  }
  v27 = a2 - v6;
  if ( a2 != v6 )
    a5 = memmove(a5, v6, a2 - v6);
  v28 = (char *)a5 + v27;
  if ( v8 != a4 )
    v28 = (char *)memmove(v28, v8, a4 - v8);
  return &v28[a4 - v8];
}
