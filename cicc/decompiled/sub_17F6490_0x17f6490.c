// Function: sub_17F6490
// Address: 0x17f6490
//
_QWORD *__fastcall sub_17F6490(__int64 a1, __int64 a2, const char *a3, __int64 a4)
{
  size_t v5; // rax
  size_t v6; // r12
  _QWORD *v7; // rdx
  _QWORD *v8; // r12
  char v9; // al
  size_t v10; // rax
  size_t v11; // r8
  _QWORD *v12; // rdx
  _QWORD *v13; // r13
  char v14; // al
  _QWORD *v16; // rdi
  __int64 v17; // rax
  _QWORD *v18; // rdi
  size_t n; // [rsp+8h] [rbp-B8h]
  _QWORD v22[2]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v23; // [rsp+40h] [rbp-80h]
  _QWORD *v24; // [rsp+50h] [rbp-70h] BYREF
  size_t v25; // [rsp+58h] [rbp-68h]
  _QWORD v26[2]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v27[2]; // [rsp+70h] [rbp-50h] BYREF
  _QWORD v28[8]; // [rsp+80h] [rbp-40h] BYREF

  v24 = v26;
  v5 = strlen(a3);
  v27[0] = v5;
  v6 = v5;
  if ( v5 > 0xF )
  {
    v24 = (_QWORD *)sub_22409D0(&v24, v27, 0);
    v16 = v24;
    v26[0] = v27[0];
  }
  else
  {
    if ( v5 == 1 )
    {
      LOBYTE(v26[0]) = *a3;
      v7 = v26;
      goto LABEL_4;
    }
    if ( !v5 )
    {
      v7 = v26;
      goto LABEL_4;
    }
    v16 = v26;
  }
  memcpy(v16, a3, v6);
  v5 = v27[0];
  v7 = v24;
LABEL_4:
  v25 = v5;
  *((_BYTE *)v7 + v5) = 0;
  if ( *(_DWORD *)(a1 + 428) == 3 )
    sub_8FD6D0((__int64)v27, byte_42B6D7A, &v24);
  else
    sub_8FD6D0((__int64)v27, "__start___", &v24);
  v22[0] = v27;
  v23 = 260;
  v8 = sub_1648A60(88, 1u);
  if ( v8 )
    sub_15E51E0((__int64)v8, a2, a4, 0, 0, 0, (__int64)v22, 0, 0, 0, 0);
  if ( (_QWORD *)v27[0] != v28 )
    j_j___libc_free_0(v27[0], v28[0] + 1LL);
  if ( v24 != v26 )
    j_j___libc_free_0(v24, v26[0] + 1LL);
  v9 = v8[4] & 0xCF | 0x10;
  *((_BYTE *)v8 + 32) = v9;
  if ( (v9 & 0xF) != 9 )
    *((_BYTE *)v8 + 33) |= 0x40u;
  v24 = v26;
  v10 = strlen(a3);
  v27[0] = v10;
  v11 = v10;
  if ( v10 > 0xF )
  {
    n = v10;
    v17 = sub_22409D0(&v24, v27, 0);
    v11 = n;
    v24 = (_QWORD *)v17;
    v18 = (_QWORD *)v17;
    v26[0] = v27[0];
  }
  else
  {
    if ( v10 == 1 )
    {
      LOBYTE(v26[0]) = *a3;
      v12 = v26;
      goto LABEL_17;
    }
    if ( !v10 )
    {
      v12 = v26;
      goto LABEL_17;
    }
    v18 = v26;
  }
  memcpy(v18, a3, v11);
  v10 = v27[0];
  v12 = v24;
LABEL_17:
  v25 = v10;
  *((_BYTE *)v12 + v10) = 0;
  if ( *(_DWORD *)(a1 + 428) == 3 )
    sub_8FD6D0((__int64)v27, byte_42B6D9E, &v24);
  else
    sub_8FD6D0((__int64)v27, "__stop___", &v24);
  v22[0] = v27;
  v23 = 260;
  v13 = sub_1648A60(88, 1u);
  if ( v13 )
    sub_15E51E0((__int64)v13, a2, a4, 0, 0, 0, (__int64)v22, 0, 0, 0, 0);
  if ( (_QWORD *)v27[0] != v28 )
    j_j___libc_free_0(v27[0], v28[0] + 1LL);
  if ( v24 != v26 )
    j_j___libc_free_0(v24, v26[0] + 1LL);
  v14 = v13[4] & 0xCF | 0x10;
  *((_BYTE *)v13 + 32) = v14;
  if ( (v14 & 0xF) != 9 )
    *((_BYTE *)v13 + 33) |= 0x40u;
  return v8;
}
