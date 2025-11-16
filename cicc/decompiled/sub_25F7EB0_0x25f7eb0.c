// Function: sub_25F7EB0
// Address: 0x25f7eb0
//
_QWORD *__fastcall sub_25F7EB0(_QWORD *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rsi
  _QWORD *v5; // r9
  __int64 v6; // rdx
  int v7; // r11d
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rbx
  bool v11; // of
  signed __int64 v12; // rbx
  _BOOL8 v13; // r13
  signed __int64 v14; // r10
  __int64 v15; // rsi
  __int64 *v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // rax
  int v19; // r8d
  __int64 v20; // rcx
  __int64 v21; // rax

  v3 = a2 - (_QWORD)a1;
  v5 = a1;
  v6 = v3 >> 3;
  if ( v3 > 0 )
  {
    v7 = 1;
    v8 = *a3;
    v9 = *(_QWORD *)(v8 + 296);
    v10 = *(_QWORD *)(v8 + 280);
    v11 = __OFSUB__(v10, v9);
    v12 = v10 - v9;
    v13 = v11;
    if ( *(_DWORD *)(v8 + 304) != 1 )
      v7 = *(_DWORD *)(v8 + 288);
    while ( 1 )
    {
      while ( 1 )
      {
        v14 = v12;
        v15 = v6 >> 1;
        v16 = &v5[v6 >> 1];
        v17 = *v16;
        if ( v13 )
        {
          v14 = 0x8000000000000000LL;
          if ( v9 <= 0 )
            v14 = 0x7FFFFFFFFFFFFFFFLL;
        }
        v18 = *(_QWORD *)(v17 + 280);
        v19 = 1;
        if ( *(_DWORD *)(v17 + 304) != 1 )
          v19 = *(_DWORD *)(v17 + 288);
        v20 = *(_QWORD *)(v17 + 296);
        v11 = __OFSUB__(v18, v20);
        v21 = v18 - v20;
        if ( v11 )
          break;
LABEL_13:
        if ( v19 != v7 )
          goto LABEL_14;
        if ( v21 > v14 )
          goto LABEL_6;
LABEL_15:
        v6 >>= 1;
        if ( v15 <= 0 )
          return v5;
      }
      if ( v20 <= 0 )
      {
        v21 = 0x7FFFFFFFFFFFFFFFLL;
        goto LABEL_13;
      }
      if ( v19 == v7 )
        goto LABEL_15;
LABEL_14:
      if ( v7 >= v19 )
        goto LABEL_15;
LABEL_6:
      v5 = v16 + 1;
      v6 = v6 - v15 - 1;
      if ( v6 <= 0 )
        return v5;
    }
  }
  return a1;
}
