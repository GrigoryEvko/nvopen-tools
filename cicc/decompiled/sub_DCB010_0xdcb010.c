// Function: sub_DCB010
// Address: 0xdcb010
//
_QWORD *__fastcall sub_DCB010(__int64 *a1, __int64 a2)
{
  __int16 v3; // ax
  void *v4; // r9
  __int64 v5; // rax
  size_t v6; // rbx
  __int64 v7; // r8
  _QWORD *v8; // rax
  __int64 v9; // rax
  _QWORD *v10; // rax
  _BYTE *v11; // rdi
  _QWORD *v12; // r12
  __int64 v14; // rax
  void *v15; // r9
  size_t v16; // r14
  __int64 v17; // r8
  __int64 *v18; // rbx
  __int64 *v19; // r14
  _QWORD *v20; // rax
  _BYTE *v21; // rdi
  _BYTE *v22; // rdi
  __int64 v23; // rax
  void *src; // [rsp+0h] [rbp-80h]
  void *srca; // [rsp+0h] [rbp-80h]
  __int64 *v26; // [rsp+8h] [rbp-78h]
  unsigned __int64 v27; // [rsp+8h] [rbp-78h]
  int v28; // [rsp+8h] [rbp-78h]
  unsigned __int64 v29; // [rsp+8h] [rbp-78h]
  int v30; // [rsp+8h] [rbp-78h]
  _BYTE *v31; // [rsp+10h] [rbp-70h] BYREF
  __int64 v32; // [rsp+18h] [rbp-68h]
  _BYTE v33[96]; // [rsp+20h] [rbp-60h] BYREF

  v3 = *(_WORD *)(a2 + 24);
  if ( v3 == 8 )
  {
    v4 = *(void **)(a2 + 32);
    v32 = 0x600000000LL;
    v5 = *(_QWORD *)(a2 + 40);
    v31 = v33;
    v6 = 8 * v5;
    v7 = (8 * v5) >> 3;
    if ( (unsigned __int64)(8 * v5) > 0x30 )
    {
      src = v4;
      v27 = (8 * v5) >> 3;
      sub_C8D5F0((__int64)&v31, v33, v27, 8u, v7, (__int64)v4);
      LODWORD(v7) = v27;
      v4 = src;
      v21 = &v31[8 * (unsigned int)v32];
    }
    else
    {
      v8 = v33;
      if ( !v6 )
        goto LABEL_4;
      v21 = v33;
    }
    v28 = v7;
    memcpy(v21, v4, v6);
    v8 = v31;
    LODWORD(v6) = v32;
    LODWORD(v7) = v28;
LABEL_4:
    LODWORD(v32) = v6 + v7;
    v9 = sub_DCB010(a1, *v8);
    *(_QWORD *)v31 = v9;
    v10 = sub_DBFF60((__int64)a1, (unsigned int *)&v31, *(_QWORD *)(a2 + 48), 0);
    v11 = v31;
    v12 = v10;
    if ( v31 == v33 )
      return v12;
LABEL_5:
    _libc_free(v11, &v31);
    return v12;
  }
  if ( v3 == 5 )
  {
    v14 = *(_QWORD *)(a2 + 40);
    v15 = *(void **)(a2 + 32);
    v31 = v33;
    v16 = 8 * v14;
    v32 = 0x600000000LL;
    v17 = (8 * v14) >> 3;
    if ( (unsigned __int64)(8 * v14) > 0x30 )
    {
      srca = v15;
      v29 = (8 * v14) >> 3;
      sub_C8D5F0((__int64)&v31, v33, v29, 8u, v17, (__int64)v15);
      LODWORD(v17) = v29;
      v15 = srca;
      v22 = &v31[8 * (unsigned int)v32];
    }
    else
    {
      v18 = (__int64 *)v33;
      if ( !v16 )
        goto LABEL_10;
      v22 = v33;
    }
    v30 = v17;
    memcpy(v22, v15, v16);
    v18 = (__int64 *)v31;
    LODWORD(v16) = v32;
    LODWORD(v17) = v30;
LABEL_10:
    LODWORD(v32) = v17 + v16;
    v26 = &v18[(unsigned int)(v17 + v16)];
    if ( v26 == v18 )
      BUG();
    v19 = 0;
    do
    {
      if ( *(_BYTE *)(sub_D95540(*v18) + 8) == 14 )
        v19 = v18;
      ++v18;
    }
    while ( v26 != v18 );
    *v19 = sub_DCB010(a1, *v19);
    v20 = sub_DC7EB0(a1, (__int64)&v31, 0, 0);
    v11 = v31;
    v12 = v20;
    if ( v31 == v33 )
      return v12;
    goto LABEL_5;
  }
  v23 = sub_D95540(a2);
  return sub_DA2C50((__int64)a1, v23, 0, 0);
}
