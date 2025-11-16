// Function: sub_22F5220
// Address: 0x22f5220
//
_DWORD *__fastcall sub_22F5220(
        _DWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  __int64 v8; // rsi
  _DWORD *v9; // r14
  __int64 v10; // r12
  __int64 v11; // r13
  __int64 v12; // rsi
  _DWORD *v13; // rbx
  unsigned int v14; // r9d
  size_t v15; // r15
  char *v16; // rdi
  size_t v17; // rax
  const char *v19; // rdi
  unsigned int v20; // [rsp+20h] [rbp-50h]

  v8 = a2 - (_QWORD)a1;
  v9 = a1;
  v10 = 0xCCCCCCCCCCCCCCCDLL * (v8 >> 4);
  if ( v8 > 0 )
  {
    while ( 1 )
    {
      v11 = v10 >> 1;
      v12 = *a7;
      v13 = &v9[20 * (v10 >> 1)];
      v14 = v13[1];
      if ( !*v13 )
        break;
      v19 = (const char *)(v12 + *(unsigned int *)(a8 + 4LL * (unsigned int)(*v13 + 1)));
      if ( !v19 )
      {
        v15 = 0;
        v16 = (char *)(v12 + v14);
        if ( !v16 )
          goto LABEL_8;
        goto LABEL_6;
      }
      v20 = v13[1];
      v15 = (unsigned int)strlen(v19);
      v17 = 0;
      v16 = (char *)(v12 + v20);
      if ( v16 )
        goto LABEL_6;
LABEL_7:
      if ( v15 > v17 )
      {
        v16 += v17;
        goto LABEL_9;
      }
LABEL_8:
      v16 += v15;
LABEL_9:
      if ( (int)sub_23C6290(v16) < 0 )
      {
        v9 = v13 + 20;
        v10 = v10 - v11 - 1;
        if ( v10 <= 0 )
          return v9;
      }
      else
      {
        v10 >>= 1;
        if ( v11 <= 0 )
          return v9;
      }
    }
    v15 = 0;
    v16 = (char *)(v12 + v14);
    if ( !v16 )
      goto LABEL_9;
LABEL_6:
    v17 = strlen(v16);
    goto LABEL_7;
  }
  return v9;
}
