// Function: sub_2651630
// Address: 0x2651630
//
__int64 __fastcall sub_2651630(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // r15
  __int64 v6; // rbx
  __int64 *v8; // r9
  char *v9; // rcx
  char *v10; // rdi
  __int64 v11; // r13
  signed __int64 v12; // r8
  const void *v13; // rsi
  unsigned __int64 v14; // rax
  _QWORD *v16; // rdx
  char *v17; // rax
  int v18; // eax
  bool v19; // al
  __int64 *v20; // [rsp+0h] [rbp-40h]
  __int64 *v21; // [rsp+0h] [rbp-40h]
  __int64 v22; // [rsp+8h] [rbp-38h] BYREF

  v4 = a2 - a1;
  v5 = a1;
  v6 = 0x8E38E38E38E38E39LL * (v4 >> 3);
  v22 = a4;
  if ( v4 > 0 )
  {
    v8 = &v22;
    do
    {
      while ( 1 )
      {
        v9 = *(char **)(a3 + 16);
        v10 = *(char **)(a3 + 8);
        v11 = v5 + 72 * (v6 >> 1);
        v12 = v9 - v10;
        v13 = *(const void **)(v11 + 8);
        v14 = *(_QWORD *)(v11 + 16) - (_QWORD)v13;
        if ( v14 >= v9 - v10 )
          break;
LABEL_6:
        v6 >>= 1;
LABEL_7:
        if ( v6 <= 0 )
          return v5;
      }
      if ( v12 == v14 )
      {
        v16 = *(_QWORD **)(v11 + 8);
        if ( v9 != v10 )
        {
          v17 = *(char **)(a3 + 8);
          while ( *(_QWORD *)v17 >= *v16 )
          {
            if ( *(_QWORD *)v17 > *v16 )
              goto LABEL_16;
            v17 += 8;
            ++v16;
            if ( v9 == v17 )
              goto LABEL_15;
          }
          v6 >>= 1;
          goto LABEL_7;
        }
LABEL_15:
        if ( *(_QWORD **)(v11 + 16) != v16 )
          goto LABEL_6;
LABEL_16:
        if ( !v12 || (v20 = v8, v18 = memcmp(v10, v13, *(_QWORD *)(a3 + 16) - (_QWORD)v10), v8 = v20, !v18) )
        {
          v21 = v8;
          v19 = sub_2650CE0(v8, a3, v5 + 72 * (v6 >> 1));
          v8 = v21;
          if ( v19 )
          {
            v6 >>= 1;
            goto LABEL_7;
          }
        }
      }
      v5 = v11 + 72;
      v6 = v6 - (v6 >> 1) - 1;
    }
    while ( v6 > 0 );
  }
  return v5;
}
