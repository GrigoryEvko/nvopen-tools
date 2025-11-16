// Function: sub_A7B660
// Address: 0xa7b660
//
unsigned __int64 __fastcall sub_A7B660(__int64 *a1, __int64 *a2, _DWORD *a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned __int64 *v10; // rdi
  const void *v11; // r9
  signed __int64 v12; // r11
  int v13; // eax
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // r8
  int v17; // r13d
  unsigned __int64 *v18; // rsi
  _DWORD *v19; // r15
  __int64 v20; // r13
  __int64 v21; // rax
  unsigned __int64 v22; // r12
  __int64 v24; // rdx
  unsigned int v25; // r13d
  unsigned __int64 *v26; // rax
  unsigned __int64 *v27; // rdx
  signed __int64 v28; // [rsp+8h] [rbp-E8h]
  const void *v29; // [rsp+10h] [rbp-E0h]
  _DWORD *v31; // [rsp+28h] [rbp-C8h]
  unsigned __int64 *v32; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v33; // [rsp+38h] [rbp-B8h]
  _BYTE dest[32]; // [rsp+40h] [rbp-B0h] BYREF
  char v35[8]; // [rsp+60h] [rbp-90h] BYREF
  char *v36; // [rsp+68h] [rbp-88h]
  char v37; // [rsp+78h] [rbp-78h] BYREF

  v8 = sub_A74460(a1);
  v9 = sub_A74450(a1);
  v10 = (unsigned __int64 *)dest;
  v11 = (const void *)v9;
  v12 = v8 - v9;
  v33 = 0x400000000LL;
  v13 = 0;
  v32 = (unsigned __int64 *)dest;
  v14 = v12 >> 3;
  if ( (unsigned __int64)v12 > 0x20 )
  {
    v28 = v12;
    v29 = v11;
    sub_C8D5F0(&v32, dest, v12 >> 3, 8);
    v13 = v33;
    v12 = v28;
    v11 = v29;
    v10 = &v32[(unsigned int)v33];
  }
  if ( (const void *)v8 != v11 )
  {
    memcpy(v10, v11, v12);
    v13 = v33;
  }
  LODWORD(v15) = v14 + v13;
  LODWORD(v33) = v15;
  v16 = a4;
  v17 = a3[a4 - 1];
  if ( v17 + 2 < (unsigned int)v15 )
    goto LABEL_6;
  v15 = (unsigned int)v15;
  v25 = v17 + 3;
  v24 = v25;
  if ( v25 == (unsigned __int64)(unsigned int)v15 )
    goto LABEL_6;
  if ( v25 < (unsigned __int64)(unsigned int)v15 )
  {
    LODWORD(v33) = v25;
LABEL_6:
    v18 = v32;
    goto LABEL_7;
  }
  if ( v25 > (unsigned __int64)HIDWORD(v33) )
  {
    sub_C8D5F0(&v32, dest, v25, 8);
    v15 = (unsigned int)v33;
    v16 = a4;
    v24 = v25;
  }
  v18 = v32;
  v26 = &v32[v15];
  v27 = &v32[v24];
  if ( v26 != v27 )
  {
    do
    {
      if ( v26 )
        *v26 = 0;
      ++v26;
    }
    while ( v27 != v26 );
    v18 = v32;
  }
  LODWORD(v33) = v25;
LABEL_7:
  v31 = &a3[v16];
  if ( &a3[v16] != a3 )
  {
    v19 = a3;
    do
    {
      v20 = (unsigned int)(*v19 + 2);
      sub_A74940((__int64)v35, (__int64)a2, v18[v20]);
      sub_A77670((__int64)v35, a5);
      v21 = sub_A7A280(a2, (__int64)v35);
      v32[v20] = v21;
      if ( v36 != &v37 )
        _libc_free(v36, v35);
      v18 = v32;
      ++v19;
    }
    while ( v31 != v19 );
  }
  v22 = sub_A77EC0(a2, v18, (unsigned int)v33);
  if ( v32 != (unsigned __int64 *)dest )
    _libc_free(v32, v18);
  return v22;
}
