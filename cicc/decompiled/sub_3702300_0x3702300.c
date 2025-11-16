// Function: sub_3702300
// Address: 0x3702300
//
unsigned __int64 *__fastcall sub_3702300(unsigned __int64 *a1, __int64 a2, unsigned __int64 *a3)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // r13
  unsigned int (*v6)(void); // rax
  __int16 v8; // ax
  unsigned __int64 v9; // rax
  __int64 v10; // rsi
  unsigned __int64 v11; // rbx
  unsigned __int32 v12; // r14d
  unsigned int (*v13)(void); // rax
  __int64 v14; // rcx
  __int16 v16; // ax
  unsigned __int64 v17; // rbx
  __int16 v18; // r14
  unsigned int (*v19)(void); // rax
  __int16 v20; // bx
  __int16 v21; // ax
  __int64 v22; // rsi
  unsigned __int64 v23; // rbx
  __int64 (*v24)(void); // rax
  __int16 v25; // r13
  unsigned int v26; // r8d
  unsigned int v27; // r8d
  unsigned int v28; // r8d
  unsigned __int32 v29; // ebx
  int v30; // r9d
  unsigned __int64 v31; // rax
  __int16 v32; // bx
  __int64 v33; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v34[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a2 + 48);
  v5 = *a3;
  v6 = *(unsigned int (**)(void))(**(_QWORD **)(v4 + 24) + 16LL);
  if ( *a3 <= 0x7FFF )
  {
    v20 = *a3;
    if ( (char *)v6 != (char *)sub_3700C70 )
    {
      v25 = __ROL2__(v5, 8);
      if ( v6() != 1 )
        v20 = v25;
    }
    LOWORD(v33) = v20;
    sub_3719260(v34, v4, &v33, 2);
    v9 = v34[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v34[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_10;
    goto LABEL_19;
  }
  if ( v5 <= 0xFFFF )
  {
    if ( (char *)v6 == (char *)sub_3700C70 || (v28 = v6(), v16 = 640, v28 == 1) )
      v16 = -32766;
    LOWORD(v33) = v16;
    sub_3719260(v34, v4, &v33, 2);
    v9 = v34[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v34[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_10;
    v10 = *(_QWORD *)(a2 + 48);
    v17 = *a3;
    v18 = v17;
    v19 = *(unsigned int (**)(void))(**(_QWORD **)(v10 + 24) + 16LL);
    if ( (char *)v19 != (char *)sub_3700C70 )
    {
      v32 = __ROL2__(v17, 8);
      if ( v19() != 1 )
        v18 = v32;
    }
    LOWORD(v33) = v18;
    v14 = 2;
    goto LABEL_9;
  }
  if ( v5 > 0xFFFFFFFF )
  {
    if ( (char *)v6 == (char *)sub_3700C70 || (v26 = v6(), v21 = 2688, v26 == 1) )
      v21 = -32758;
    LOWORD(v33) = v21;
    sub_3719260(v34, v4, &v33, 2);
    v9 = v34[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v34[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_10;
    v22 = *(_QWORD *)(a2 + 48);
    v23 = *a3;
    v24 = *(__int64 (**)(void))(**(_QWORD **)(v22 + 24) + 16LL);
    if ( v24 != sub_3700C70 )
    {
      v30 = v24();
      v31 = _byteswap_uint64(v23);
      if ( v30 != 1 )
        v23 = v31;
    }
    v34[0] = v23;
    sub_3719260(&v33, v22, v34, 8);
    v9 = v33 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v33 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_10;
LABEL_19:
    *a1 = 1;
    return a1;
  }
  if ( (char *)v6 == (char *)sub_3700C70 || (v27 = v6(), v8 = 1152, v27 == 1) )
    v8 = -32764;
  LOWORD(v33) = v8;
  sub_3719260(v34, v4, &v33, 2);
  v9 = v34[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v34[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_10;
  v10 = *(_QWORD *)(a2 + 48);
  v11 = *a3;
  v12 = v11;
  v13 = *(unsigned int (**)(void))(**(_QWORD **)(v10 + 24) + 16LL);
  if ( (char *)v13 != (char *)sub_3700C70 )
  {
    v29 = _byteswap_ulong(v11);
    if ( v13() != 1 )
      v12 = v29;
  }
  LODWORD(v33) = v12;
  v14 = 4;
LABEL_9:
  sub_3719260(v34, v10, &v33, v14);
  v9 = v34[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v34[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    goto LABEL_19;
LABEL_10:
  *a1 = v9 | 1;
  return a1;
}
