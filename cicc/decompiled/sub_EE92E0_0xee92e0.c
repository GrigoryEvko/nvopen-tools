// Function: sub_EE92E0
// Address: 0xee92e0
//
__int64 __fastcall sub_EE92E0(_QWORD **a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r14d
  __int64 v7; // rcx
  _DWORD *v8; // rbx
  char v9; // al
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  _QWORD *v18; // rax
  __int64 v19; // r15
  __int64 *v20; // rsi
  __int64 *v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  __int64 **v25; // rbx
  __int64 *v26; // r12
  __int64 v28; // rax
  __int64 *v29; // rdx
  __int64 *v30; // r14
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 *v33; // rcx
  __int64 *v34; // rax
  __int64 *v35; // rax
  char v37; // [rsp+Fh] [rbp-E1h]
  __int64 n; // [rsp+18h] [rbp-D8h]
  __int64 *v39; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v40[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v41[176]; // [rsp+40h] [rbp-B0h] BYREF

  v6 = *((_DWORD *)*a1 + a2 + 198);
  v7 = v6 + 1;
  *((_DWORD *)*a1 + a2 + 198) = v7;
  v8 = *a1;
  v9 = *((_BYTE *)*a1 + 937);
  v40[0] = (__int64)v41;
  v37 = v9;
  v40[1] = 0x2000000000LL;
  sub_D953B0((__int64)v40, 33, a2 + 196LL, v7, a5, a6);
  sub_D953B0((__int64)v40, a2, v10, v11, v12, v13);
  sub_D953B0((__int64)v40, v6, v14, v15, v16, v17);
  v18 = sub_C65B40((__int64)(v8 + 226), (__int64)v40, (__int64 *)&v39, (__int64)off_497B2F0);
  v19 = (__int64)v18;
  if ( v18 )
  {
    v19 = (__int64)(v18 + 1);
    if ( (_BYTE *)v40[0] != v41 )
      _libc_free(v40[0], v40);
    v20 = v40;
    v40[0] = v19;
    v21 = sub_EE6840((__int64)(v8 + 236), v40);
    if ( v21 )
    {
      v24 = v21[1];
      if ( v24 )
        v19 = v24;
    }
    if ( *((_QWORD *)v8 + 116) == v19 )
      *((_BYTE *)v8 + 936) = 1;
  }
  else
  {
    if ( !v37 )
    {
      if ( (_BYTE *)v40[0] != v41 )
        _libc_free(v40[0], v40);
      *((_QWORD *)v8 + 115) = 0;
      return v19;
    }
    v28 = sub_CD1D40((__int64 *)v8 + 101, 32, 3);
    *(_QWORD *)v28 = 0;
    v20 = (__int64 *)v28;
    v19 = v28 + 8;
    *(_WORD *)(v28 + 16) = 16417;
    LOBYTE(v28) = *(_BYTE *)(v28 + 18);
    *((_DWORD *)v20 + 6) = v6;
    v29 = v39;
    *((_BYTE *)v20 + 18) = v28 & 0xF0 | 5;
    v20[1] = (__int64)&unk_49DFA28;
    *((_DWORD *)v20 + 5) = a2;
    sub_C657C0((__int64 *)v8 + 113, v20, v29, (__int64)off_497B2F0);
    if ( (_BYTE *)v40[0] != v41 )
      _libc_free(v40[0], v20);
    *((_QWORD *)v8 + 115) = v19;
  }
  v25 = (__int64 **)*a1[1];
  if ( v25 )
  {
    v26 = v25[1];
    if ( v26 != v25[2] )
    {
LABEL_11:
      v25[1] = v26 + 1;
      *v26 = v19;
      return v19;
    }
    v30 = *v25;
    n = (char *)v26 - (char *)*v25;
    if ( *v25 == (__int64 *)(v25 + 3) )
    {
      v34 = (__int64 *)malloc(16 * (n >> 3), v20, n, 16 * (n >> 3), v22, v23);
      v33 = v34;
      if ( v34 )
      {
        v32 = n;
        if ( v26 != v30 )
        {
          v35 = (__int64 *)memmove(v34, v30, n);
          v32 = n;
          v33 = v35;
        }
        *v25 = v33;
        goto LABEL_23;
      }
    }
    else
    {
      v31 = realloc(v30);
      v32 = n;
      *v25 = (__int64 *)v31;
      v33 = (__int64 *)v31;
      if ( v31 )
      {
LABEL_23:
        v26 = (__int64 *)((char *)v33 + v32);
        v25[2] = &v33[2 * (n >> 3)];
        goto LABEL_11;
      }
    }
    abort();
  }
  return v19;
}
