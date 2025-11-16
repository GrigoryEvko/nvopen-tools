// Function: sub_C31760
// Address: 0xc31760
//
void __fastcall sub_C31760(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  bool v6; // zf
  __int64 v7; // rsi
  __int64 v8; // rax
  int v9; // edx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  char *v16; // r13
  char *v17; // rbx
  char *i; // rax
  __int64 v19; // rdi
  unsigned int v20; // ecx
  __int64 *v21; // rbx
  __int64 *v22; // r12
  __int64 v23; // rdi
  __int64 v24[5]; // [rsp+0h] [rbp-B0h] BYREF
  char *v25; // [rsp+28h] [rbp-88h] BYREF
  __int64 v26; // [rsp+30h] [rbp-80h]
  _BYTE v27[32]; // [rsp+38h] [rbp-78h] BYREF
  __int64 *v28; // [rsp+58h] [rbp-58h] BYREF
  __int64 v29; // [rsp+60h] [rbp-50h]
  _QWORD v30[3]; // [rsp+68h] [rbp-48h] BYREF
  char v31; // [rsp+80h] [rbp-30h]

  v6 = *(_BYTE *)(a4 + 128) == 0;
  v31 = 0;
  if ( !v6 )
  {
    v8 = *(_QWORD *)a4;
    v9 = *(_DWORD *)(a4 + 48);
    *(_QWORD *)a4 = 0;
    v24[0] = v8;
    v11 = *(_QWORD *)(a4 + 8);
    *(_QWORD *)(a4 + 8) = 0;
    v24[1] = v11;
    v12 = *(_QWORD *)(a4 + 16);
    *(_DWORD *)(a4 + 16) = 0;
    v24[2] = v12;
    v24[3] = *(_QWORD *)(a4 + 24);
    v24[4] = *(_QWORD *)(a4 + 32);
    v25 = v27;
    v26 = 0x400000000LL;
    if ( v9 )
      sub_C2F210((__int64)&v25, (char **)(a4 + 40));
    v29 = 0;
    v28 = v30;
    if ( *(_DWORD *)(a4 + 96) )
      sub_C2F080((__int64 *)&v28, a4 + 88);
    v13 = *(_QWORD *)(a4 + 104);
    *(_QWORD *)(a4 + 32) = 0;
    *(_QWORD *)(a4 + 24) = 0;
    v30[0] = v13;
    v14 = *(_QWORD *)(a4 + 112);
    *(_QWORD *)(a4 + 104) = 0;
    v30[1] = v14;
    v15 = *(_QWORD *)(a4 + 120);
    *(_DWORD *)(a4 + 48) = 0;
    *(_DWORD *)(a4 + 96) = 0;
    v30[2] = v15;
    v31 = 1;
  }
  v7 = 1;
  sub_C31010(a1, 1, a2, a3, v24);
  if ( v31 )
  {
    v16 = v25;
    v31 = 0;
    v17 = &v25[8 * (unsigned int)v26];
    if ( v25 != v17 )
    {
      for ( i = v25; ; i = v25 )
      {
        v19 = *(_QWORD *)v16;
        v20 = (unsigned int)((v16 - i) >> 3) >> 7;
        v7 = 4096LL << v20;
        if ( v20 >= 0x1E )
          v7 = 0x40000000000LL;
        v16 += 8;
        sub_C7D6A0(v19, v7, 16);
        if ( v17 == v16 )
          break;
      }
    }
    v21 = v28;
    v22 = &v28[2 * (unsigned int)v29];
    if ( v28 != v22 )
    {
      do
      {
        v7 = v21[1];
        v23 = *v21;
        v21 += 2;
        sub_C7D6A0(v23, v7, 16);
      }
      while ( v22 != v21 );
      v22 = v28;
    }
    if ( v22 != v30 )
      _libc_free(v22, v7);
    if ( v25 != v27 )
      _libc_free(v25, v7);
    _libc_free(v24[0], v7);
  }
}
