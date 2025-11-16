// Function: sub_30AB9D0
// Address: 0x30ab9d0
//
void __fastcall sub_30AB9D0(__int64 a1, __int64 a2, __int64 **a3, unsigned __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v6; // r14
  bool v7; // zf
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 *v10; // r15
  __int64 **v11; // rax
  __int64 v12; // rsi
  unsigned __int64 v13[2]; // [rsp+10h] [rbp-1E0h] BYREF
  _BYTE v14[128]; // [rsp+20h] [rbp-1D0h] BYREF
  __int64 v15; // [rsp+A0h] [rbp-150h] BYREF
  char *v16; // [rsp+A8h] [rbp-148h]
  __int64 v17; // [rsp+B0h] [rbp-140h]
  int v18; // [rsp+B8h] [rbp-138h]
  char v19; // [rsp+BCh] [rbp-134h]
  char v20; // [rsp+C0h] [rbp-130h] BYREF

  v6 = (__int64)a3;
  v16 = &v20;
  v7 = *(_BYTE *)(a2 + 192) == 0;
  v13[0] = (unsigned __int64)v14;
  v15 = 0;
  v17 = 32;
  v18 = 0;
  v19 = 1;
  v13[1] = 0x1000000000LL;
  if ( v7 )
    sub_CFDFC0(a2, a2, (__int64)a3, a4, a5, a6);
  v8 = *(_QWORD *)(a2 + 16);
  v9 = v8 + 32LL * *(unsigned int *)(a2 + 24);
  if ( v9 != v8 )
  {
    while ( 1 )
    {
      v10 = *(__int64 **)(v8 + 16);
      if ( !v10 )
        goto LABEL_10;
      if ( *(_BYTE *)(v6 + 28) )
      {
        v11 = *(__int64 ***)(v6 + 8);
        v12 = *(unsigned int *)(v6 + 20);
        a3 = &v11[v12];
        if ( v11 != a3 )
        {
          while ( v10 != *v11 )
          {
            if ( a3 == ++v11 )
              goto LABEL_18;
          }
          goto LABEL_10;
        }
LABEL_18:
        if ( (unsigned int)v12 < *(_DWORD *)(v6 + 16) )
        {
          *(_DWORD *)(v6 + 20) = v12 + 1;
          *a3 = v10;
          ++*(_QWORD *)v6;
          goto LABEL_16;
        }
      }
      sub_C8CC70(v6, *(_QWORD *)(v8 + 16), (__int64)a3, a4, a5, (__int64)a6);
      if ( (_BYTE)a3 )
      {
LABEL_16:
        v8 += 32;
        sub_30AB500(v10, (__int64)&v15, (__int64)v13, a4, a5, (__int64)a6);
        if ( v9 == v8 )
          break;
      }
      else
      {
LABEL_10:
        v8 += 32;
        if ( v9 == v8 )
          break;
      }
    }
  }
  sub_30AB660((__int64)&v15, (__int64)v13, (__int64 **)v6, a4, a5, (__int64)a6);
  if ( (_BYTE *)v13[0] != v14 )
    _libc_free(v13[0]);
  if ( !v19 )
    _libc_free((unsigned __int64)v16);
}
