// Function: sub_1BE27E0
// Address: 0x1be27e0
//
void *__fastcall sub_1BE27E0(__int64 a1, __int64 a2)
{
  int v3; // edi
  char *v4; // rax
  char *v5; // r15
  size_t v6; // rax
  _BYTE *v7; // rdi
  size_t v8; // rdx
  _BYTE *v9; // rax
  void **v10; // r9
  unsigned int v11; // r15d
  void *result; // rax
  _WORD *v13; // rdx
  __int64 *v14; // rax
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // r10
  void **v19; // rsi
  __int64 v20; // rax
  __int64 v21; // [rsp+8h] [rbp-B8h]
  size_t v23; // [rsp+18h] [rbp-A8h]
  __int64 v24; // [rsp+18h] [rbp-A8h]
  _QWORD v25[2]; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD v26[2]; // [rsp+30h] [rbp-90h] BYREF
  char *v27[2]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v28; // [rsp+50h] [rbp-70h] BYREF
  void *v29; // [rsp+60h] [rbp-60h] BYREF
  void *v30; // [rsp+68h] [rbp-58h]
  _BYTE *v31; // [rsp+70h] [rbp-50h]
  void *dest; // [rsp+78h] [rbp-48h]
  int v33; // [rsp+80h] [rbp-40h]
  _QWORD *v34; // [rsp+88h] [rbp-38h]

  v3 = *(unsigned __int8 *)(a2 + 16);
  v25[0] = v26;
  LOBYTE(v26[0]) = 0;
  v25[1] = 0;
  v33 = 1;
  dest = 0;
  v31 = 0;
  v30 = 0;
  v29 = &unk_49EFBE0;
  v34 = v25;
  if ( (unsigned __int8)v3 <= 0x17u )
  {
    sub_15537D0(a2, (__int64)&v29, 0, 0);
    goto LABEL_8;
  }
  if ( !*(_BYTE *)(*(_QWORD *)a2 + 8LL) )
  {
    v4 = sub_15F29F0(v3 - 24);
    v5 = v4;
    if ( v4 )
      goto LABEL_4;
LABEL_18:
    v9 = v31;
    v7 = dest;
    v10 = &v29;
    goto LABEL_19;
  }
  sub_15537D0(a2, (__int64)&v29, 0, 0);
  v13 = dest;
  if ( (unsigned __int64)(v31 - (_BYTE *)dest) <= 2 )
  {
    sub_16E7EE0((__int64)&v29, " = ", 3u);
  }
  else
  {
    *((_BYTE *)dest + 2) = 32;
    *v13 = 15648;
    dest = (char *)dest + 3;
  }
  v4 = sub_15F29F0((unsigned int)*(unsigned __int8 *)(a2 + 16) - 24);
  v5 = v4;
  if ( !v4 )
    goto LABEL_18;
LABEL_4:
  v6 = strlen(v4);
  v7 = dest;
  v8 = v6;
  v9 = v31;
  if ( v8 > v31 - (_BYTE *)dest )
  {
    v10 = (void **)sub_16E7EE0((__int64)&v29, v5, v8);
    v9 = v10[2];
    v7 = v10[3];
  }
  else
  {
    v10 = &v29;
    if ( v8 )
    {
      v23 = v8;
      memcpy(dest, v5, v8);
      v10 = &v29;
      dest = (char *)dest + v23;
      v7 = dest;
      if ( dest == v31 )
        goto LABEL_7;
LABEL_20:
      *v7 = 32;
      v10[3] = (char *)v10[3] + 1;
      v11 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      if ( !v11 )
        goto LABEL_8;
      goto LABEL_21;
    }
  }
LABEL_19:
  if ( v7 != v9 )
    goto LABEL_20;
LABEL_7:
  sub_16E7EE0((__int64)v10, " ", 1u);
  v11 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( !v11 )
    goto LABEL_8;
LABEL_21:
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v14 = *(__int64 **)(a2 - 8);
  else
    v14 = (__int64 *)(a2 - 24LL * v11);
  sub_15537D0(*v14, (__int64)&v29, 0, 0);
  if ( v11 != 1 )
  {
    v15 = v11 - 2;
    v16 = 24;
    v24 = 24 * v15 + 48;
    do
    {
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v17 = *(_QWORD *)(a2 - 8);
      else
        v17 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v18 = *(_QWORD *)(v17 + v16);
      if ( (unsigned __int64)(v31 - (_BYTE *)dest) <= 1 )
      {
        v21 = *(_QWORD *)(v17 + v16);
        v20 = sub_16E7EE0((__int64)&v29, ", ", 2u);
        v18 = v21;
        v19 = (void **)v20;
      }
      else
      {
        v19 = &v29;
        *(_WORD *)dest = 8236;
        dest = (char *)dest + 2;
      }
      v16 += 24;
      sub_15537D0(v18, (__int64)v19, 0, 0);
    }
    while ( v24 != v16 );
  }
LABEL_8:
  if ( dest != v30 )
    sub_16E7BA0((__int64 *)&v29);
  sub_16BE9B0((__int64 *)v27, (__int64)v25);
  sub_16E7EE0(a1, v27[0], (size_t)v27[1]);
  if ( (__int64 *)v27[0] != &v28 )
    j_j___libc_free_0(v27[0], v28 + 1);
  result = sub_16E7BC0((__int64 *)&v29);
  if ( (_QWORD *)v25[0] != v26 )
    return (void *)j_j___libc_free_0(v25[0], v26[0] + 1LL);
  return result;
}
