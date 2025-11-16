// Function: sub_30AB790
// Address: 0x30ab790
//
void __fastcall sub_30AB790(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 *a6)
{
  bool v7; // zf
  __int64 v8; // r14
  __int64 v9; // r10
  __int64 v10; // r12
  __int64 *v11; // r15
  __int64 v12; // rsi
  __int64 **v13; // rax
  __int64 **v14; // rdx
  __int64 **v15; // rax
  __int64 v16; // rsi
  char v17; // dl
  unsigned __int64 v18[2]; // [rsp+20h] [rbp-1E0h] BYREF
  _BYTE v19[128]; // [rsp+30h] [rbp-1D0h] BYREF
  __int64 v20; // [rsp+B0h] [rbp-150h] BYREF
  char *v21; // [rsp+B8h] [rbp-148h]
  __int64 v22; // [rsp+C0h] [rbp-140h]
  int v23; // [rsp+C8h] [rbp-138h]
  char v24; // [rsp+CCh] [rbp-134h]
  char v25; // [rsp+D0h] [rbp-130h] BYREF

  v21 = &v25;
  v7 = *(_BYTE *)(a2 + 192) == 0;
  v18[0] = (unsigned __int64)v19;
  v20 = 0;
  v22 = 32;
  v23 = 0;
  v24 = 1;
  v18[1] = 0x1000000000LL;
  if ( v7 )
    sub_CFDFC0(a2, a2, a3, a4, a5, a6);
  v8 = *(_QWORD *)(a2 + 16);
  v9 = 32LL * *(unsigned int *)(a2 + 24);
  v10 = v8 + v9;
  if ( v8 + v9 != v8 )
  {
    while ( 1 )
    {
      v11 = *(__int64 **)(v8 + 16);
      if ( !v11 )
        goto LABEL_15;
      v12 = v11[5];
      if ( *(_BYTE *)(a1 + 84) )
      {
        v13 = *(__int64 ***)(a1 + 64);
        v14 = &v13[*(unsigned int *)(a1 + 76)];
        if ( v13 == v14 )
          goto LABEL_15;
        while ( (__int64 *)v12 != *v13 )
        {
          if ( v14 == ++v13 )
            goto LABEL_15;
        }
        if ( !*(_BYTE *)(a3 + 28) )
          goto LABEL_22;
        goto LABEL_11;
      }
      if ( sub_C8CA60(a1 + 56, v12) )
      {
        if ( !*(_BYTE *)(a3 + 28) )
          goto LABEL_22;
LABEL_11:
        v15 = *(__int64 ***)(a3 + 8);
        v16 = *(unsigned int *)(a3 + 20);
        v14 = &v15[v16];
        if ( v15 != v14 )
        {
          while ( v11 != *v15 )
          {
            if ( v14 == ++v15 )
              goto LABEL_25;
          }
          goto LABEL_15;
        }
LABEL_25:
        if ( (unsigned int)v16 < *(_DWORD *)(a3 + 16) )
        {
          *(_DWORD *)(a3 + 20) = v16 + 1;
          *v14 = v11;
          ++*(_QWORD *)a3;
          goto LABEL_23;
        }
LABEL_22:
        sub_C8CC70(a3, (__int64)v11, (__int64)v14, a4, a5, (__int64)a6);
        if ( !v17 )
          goto LABEL_15;
LABEL_23:
        v8 += 32;
        sub_30AB500(v11, (__int64)&v20, (__int64)v18, a4, a5, (__int64)a6);
        if ( v10 == v8 )
          break;
      }
      else
      {
LABEL_15:
        v8 += 32;
        if ( v10 == v8 )
          break;
      }
    }
  }
  sub_30AB660((__int64)&v20, (__int64)v18, (__int64 **)a3, a4, a5, (__int64)a6);
  if ( (_BYTE *)v18[0] != v19 )
    _libc_free(v18[0]);
  if ( !v24 )
    _libc_free((unsigned __int64)v21);
}
