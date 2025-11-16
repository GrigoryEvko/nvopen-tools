// Function: sub_3701ED0
// Address: 0x3701ed0
//
unsigned __int64 *__fastcall sub_3701ED0(unsigned __int64 *a1, __int64 a2, unsigned __int64 *a3)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // rbx
  __int64 (*v6)(void); // rax
  __int16 v8; // ax
  unsigned __int64 v9; // rax
  __int64 v10; // rsi
  unsigned __int64 v11; // r13
  __int64 (*v12)(void); // rax
  __int64 v13; // rcx
  __int16 v14; // ax
  unsigned __int64 v15; // r13
  void (*v16)(void); // rax
  __int16 v18; // ax
  __int64 v19; // rsi
  unsigned __int64 v20; // r13
  __int64 (*v21)(void); // rax
  __int16 v22; // ax
  unsigned __int64 v23; // r13
  __int64 (*v24)(void); // rax
  int v25; // r8d
  __int16 v26; // ax
  int v27; // r8d
  int v28; // r9d
  unsigned __int32 v29; // eax
  int v30; // r9d
  unsigned __int64 v31; // rax
  int v32; // r9d
  __int16 v33; // ax
  int v34; // r8d
  int v35; // r8d
  int v36; // r8d
  __int64 v37; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v38[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a2 + 48);
  v5 = *a3;
  v6 = *(__int64 (**)(void))(**(_QWORD **)(v4 + 24) + 16LL);
  if ( *a3 <= 0x7FFF )
  {
    if ( v6 != sub_3700C70 )
    {
      v25 = v6();
      v26 = __ROL2__(v5, 8);
      if ( v25 != 1 )
        LOWORD(v5) = v26;
    }
    LOWORD(v37) = v5;
    sub_3719260(v38, v4, &v37, 2);
    v9 = v38[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v38[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_17;
    goto LABEL_21;
  }
  if ( v5 + 128 <= 0xFF )
  {
    if ( v6 == sub_3700C70 || (v27 = v6(), v14 = 128, v27 == 1) )
      v14 = 0x8000;
    LOWORD(v37) = v14;
    sub_3719260(v38, v4, &v37, 2);
    v9 = v38[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v38[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_17;
    v10 = *(_QWORD *)(a2 + 48);
    v15 = *a3;
    v16 = *(void (**)(void))(**(_QWORD **)(v10 + 24) + 16LL);
    if ( (char *)v16 != (char *)sub_3700C70 )
      v16();
    LOBYTE(v37) = v15;
    v13 = 1;
    goto LABEL_16;
  }
  if ( v5 + 0x8000 <= 0xFFFF )
  {
    if ( v6 == sub_3700C70 || (v36 = v6(), v22 = 384, v36 == 1) )
      v22 = -32767;
    LOWORD(v37) = v22;
    sub_3719260(v38, v4, &v37, 2);
    v9 = v38[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v38[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_17;
    v10 = *(_QWORD *)(a2 + 48);
    v23 = *a3;
    v24 = *(__int64 (**)(void))(**(_QWORD **)(v10 + 24) + 16LL);
    if ( v24 != sub_3700C70 )
    {
      v32 = v24();
      v33 = __ROL2__(v23, 8);
      if ( v32 != 1 )
        LOWORD(v23) = v33;
    }
    LOWORD(v37) = v23;
    v13 = 2;
    goto LABEL_16;
  }
  if ( v5 + 0x80000000 > 0xFFFFFFFF )
  {
    if ( v6 == sub_3700C70 || (v35 = v6(), v18 = 2432, v35 == 1) )
      v18 = -32759;
    LOWORD(v37) = v18;
    sub_3719260(v38, v4, &v37, 2);
    v9 = v38[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v38[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_17;
    v19 = *(_QWORD *)(a2 + 48);
    v20 = *a3;
    v21 = *(__int64 (**)(void))(**(_QWORD **)(v19 + 24) + 16LL);
    if ( v21 != sub_3700C70 )
    {
      v30 = v21();
      v31 = _byteswap_uint64(v20);
      if ( v30 != 1 )
        v20 = v31;
    }
    v38[0] = v20;
    sub_3719260(&v37, v19, v38, 8);
    v9 = v37 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v37 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_17;
LABEL_21:
    *a1 = 1;
    return a1;
  }
  if ( v6 == sub_3700C70 || (v34 = v6(), v8 = 896, v34 == 1) )
    v8 = -32765;
  LOWORD(v37) = v8;
  sub_3719260(v38, v4, &v37, 2);
  v9 = v38[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v38[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_17;
  v10 = *(_QWORD *)(a2 + 48);
  v11 = *a3;
  v12 = *(__int64 (**)(void))(**(_QWORD **)(v10 + 24) + 16LL);
  if ( v12 != sub_3700C70 )
  {
    v28 = v12();
    v29 = _byteswap_ulong(v11);
    if ( v28 != 1 )
      LODWORD(v11) = v29;
  }
  LODWORD(v37) = v11;
  v13 = 4;
LABEL_16:
  sub_3719260(v38, v10, &v37, v13);
  v9 = v38[0] & 0xFFFFFFFFFFFFFFFELL;
  if ( (v38[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    goto LABEL_21;
LABEL_17:
  *a1 = v9 | 1;
  return a1;
}
