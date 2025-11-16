// Function: sub_29F7E10
// Address: 0x29f7e10
//
void __fastcall sub_29F7E10(__int64 a1, unsigned __int64 a2, unsigned __int64 a3, __int64 a4, char *a5, __int64 a6)
{
  unsigned __int64 v6; // r14
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 i; // r13
  bool v11; // zf
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rsi
  char *v15; // rax
  __int64 v16; // rdi
  char *v17; // r9
  __int64 v18; // rcx
  char *v19; // rdx
  __int64 v20; // rdx
  char *v21; // r8
  __int64 v22; // r9
  __int64 v23; // [rsp+8h] [rbp-118h]
  char *v25; // [rsp+20h] [rbp-100h]
  char *v26; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v27; // [rsp+38h] [rbp-E8h]
  __int64 v28; // [rsp+40h] [rbp-E0h]
  _BYTE v29[72]; // [rsp+48h] [rbp-D8h] BYREF
  _BYTE *v30; // [rsp+90h] [rbp-90h] BYREF
  __int64 v31; // [rsp+98h] [rbp-88h]
  __int64 v32; // [rsp+A0h] [rbp-80h]
  _BYTE v33[120]; // [rsp+A8h] [rbp-78h] BYREF

  v6 = a2;
  v23 = a2 - a1;
  if ( (__int64)(a2 - a1) > 88 )
  {
    v8 = 0x2E8BA2E8BA2E8BA3LL * ((__int64)(a2 - a1) >> 3);
    v25 = (char *)v8;
    v9 = (v8 - 2) / 2;
    for ( i = a1 + 88 * v9; ; i -= 88 )
    {
      v11 = *(_QWORD *)(i + 8) == 0;
      v27 = 0;
      v28 = 64;
      v26 = v29;
      if ( v11 )
      {
        v30 = v33;
        v31 = 0;
        v32 = 64;
      }
      else
      {
        sub_29F3DD0((__int64)&v26, (char **)i, v8, a4, (__int64)a5, a6);
        v30 = v33;
        v31 = 0;
        v32 = 64;
        if ( v27 )
          sub_29F3DD0((__int64)&v30, &v26, v12, v13, (__int64)a5, a6);
      }
      sub_29F4C70(a1, v9, v25, (__int64)&v30, a5, a6);
      if ( v30 != v33 )
        _libc_free((unsigned __int64)v30);
      if ( !v9 )
        break;
      --v9;
      if ( v26 != v29 )
        _libc_free((unsigned __int64)v26);
    }
    v6 = a2;
    if ( v26 != v29 )
      _libc_free((unsigned __int64)v26);
  }
  if ( v6 < a3 )
  {
    while ( 1 )
    {
      v14 = *(_QWORD *)(a1 + 8);
      v15 = *(char **)v6;
      v16 = *(_QWORD *)(v6 + 8);
      v17 = *(char **)a1;
      v18 = *(_QWORD *)v6 + v14;
      if ( v16 <= v14 )
        v18 = *(_QWORD *)v6 + v16;
      v19 = *(char **)a1;
      if ( v15 != (char *)v18 )
        break;
LABEL_33:
      if ( v19 != &v17[v14] )
        goto LABEL_23;
LABEL_31:
      v6 += 88LL;
      if ( a3 <= v6 )
        return;
    }
    while ( *v15 >= *v19 )
    {
      if ( *v15 > *v19 )
        goto LABEL_31;
      ++v15;
      ++v19;
      if ( (char *)v18 == v15 )
        goto LABEL_33;
    }
LABEL_23:
    v27 = 0;
    v26 = v29;
    v28 = 64;
    if ( v16 )
      sub_29F3DD0((__int64)&v26, (char **)v6, (__int64)v19, v18, (__int64)a5, (__int64)v17);
    sub_29F3DD0(v6, (char **)a1, (__int64)v19, v18, (__int64)a5, (__int64)v17);
    v30 = v33;
    v31 = 0;
    v32 = 64;
    if ( v27 )
      sub_29F3DD0((__int64)&v30, &v26, v20, (__int64)&v30, (__int64)v21, v22);
    sub_29F4C70(a1, 0, (char *)(0x2E8BA2E8BA2E8BA3LL * (v23 >> 3)), (__int64)&v30, v21, v22);
    if ( v30 != v33 )
      _libc_free((unsigned __int64)v30);
    if ( v26 != v29 )
      _libc_free((unsigned __int64)v26);
    goto LABEL_31;
  }
}
