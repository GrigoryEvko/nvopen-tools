// Function: sub_25F7DB0
// Address: 0x25f7db0
//
__int64 *__fastcall sub_25F7DB0(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rsi
  __int64 *v5; // r9
  __int64 v6; // rdx
  int v7; // r10d
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r12
  bool v11; // of
  __int64 v12; // r12
  _BOOL8 v13; // rbx
  __int64 v14; // rcx
  int v15; // r8d
  __int64 v16; // rsi
  __int64 *v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rcx
  signed __int64 v21; // rax

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
      v15 = 1;
      v16 = v6 >> 1;
      v17 = &v5[v6 >> 1];
      v18 = *v17;
      v19 = *(_QWORD *)(*v17 + 280);
      if ( *(_DWORD *)(*v17 + 304) != 1 )
        v15 = *(_DWORD *)(v18 + 288);
      v20 = *(_QWORD *)(v18 + 296);
      v11 = __OFSUB__(v19, v20);
      v21 = v19 - v20;
      if ( v11 )
      {
        v21 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v20 > 0 )
          v21 = 0x8000000000000000LL;
      }
      if ( v13 )
      {
        v14 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v9 > 0 )
        {
          if ( v7 == v15 )
            goto LABEL_16;
          goto LABEL_7;
        }
      }
      else
      {
        v14 = v12;
      }
      if ( v7 == v15 )
      {
        if ( v14 <= v21 )
          goto LABEL_16;
LABEL_8:
        v6 >>= 1;
        if ( v16 <= 0 )
          return v5;
      }
      else
      {
LABEL_7:
        if ( v15 < v7 )
          goto LABEL_8;
LABEL_16:
        v5 = v17 + 1;
        v6 = v6 - v16 - 1;
        if ( v6 <= 0 )
          return v5;
      }
    }
  }
  return a1;
}
