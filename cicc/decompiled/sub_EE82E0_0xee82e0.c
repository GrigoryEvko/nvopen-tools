// Function: sub_EE82E0
// Address: 0xee82e0
//
_QWORD *__fastcall sub_EE82E0(__int64 a1, __int64 *a2, char *a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v8; // al
  __int64 v9; // rsi
  size_t v10; // rax
  __int64 *v11; // rsi
  _QWORD *v12; // rax
  _QWORD *v13; // r8
  __int64 v14; // r8
  __int64 *v15; // rax
  __int64 v16; // rax
  __int64 *v18; // rax
  __int64 *v19; // r13
  __int64 v20; // r14
  size_t v21; // rax
  size_t v22; // r10
  __int64 *v23; // rdx
  char v24; // [rsp+0h] [rbp-E0h]
  _QWORD *v25; // [rsp+8h] [rbp-D8h]
  __int64 v26; // [rsp+8h] [rbp-D8h]
  _QWORD *v27; // [rsp+8h] [rbp-D8h]
  __int64 *v28; // [rsp+18h] [rbp-C8h] BYREF
  __int64 v29[2]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD v30[22]; // [rsp+30h] [rbp-B0h] BYREF

  v8 = *(_BYTE *)(a1 + 129);
  v9 = *a2;
  v29[0] = (__int64)v30;
  v24 = v8;
  v29[1] = 0x2000000002LL;
  v30[0] = 5;
  sub_D953B0((__int64)v29, v9, (__int64)a3, a4, a5, a6);
  v10 = strlen(a3);
  if ( v10 )
    sub_C653C0((__int64)v29, (unsigned __int8 *)a3, v10);
  else
    sub_C653C0((__int64)v29, 0, 0);
  v11 = v29;
  v12 = sub_C65B40(a1 + 96, (__int64)v29, (__int64 *)&v28, (__int64)off_497B2F0);
  v13 = v12;
  if ( v12 )
  {
    v14 = (__int64)(v12 + 1);
    if ( (_QWORD *)v29[0] != v30 )
    {
      v25 = v12 + 1;
      _libc_free(v29[0], v29);
      v14 = (__int64)v25;
    }
    v29[0] = v14;
    v26 = v14;
    v15 = sub_EE6840(a1 + 136, v29);
    v13 = (_QWORD *)v26;
    if ( v15 )
    {
      v16 = v15[1];
      if ( v16 )
        v13 = (_QWORD *)v16;
    }
    if ( *(_QWORD **)(a1 + 120) == v13 )
      *(_BYTE *)(a1 + 128) = 1;
  }
  else
  {
    if ( v24 )
    {
      v18 = (__int64 *)sub_CD1D40((__int64 *)a1, 48, 3);
      *v18 = 0;
      v19 = v18;
      v20 = *a2;
      v21 = strlen(a3);
      v19[3] = v20;
      v11 = v19;
      v22 = v21;
      v19[5] = (__int64)a3;
      *((_WORD *)v19 + 8) = 16389;
      LOBYTE(v21) = *((_BYTE *)v19 + 18);
      v19[4] = v22;
      v23 = v28;
      *((_BYTE *)v19 + 18) = v21 & 0xF0 | 5;
      v19[1] = (__int64)&unk_49DEF48;
      sub_C657C0((__int64 *)(a1 + 96), v19, v23, (__int64)off_497B2F0);
      v13 = v19 + 1;
    }
    if ( (_QWORD *)v29[0] != v30 )
    {
      v27 = v13;
      _libc_free(v29[0], v11);
      v13 = v27;
    }
    *(_QWORD *)(a1 + 112) = v13;
  }
  return v13;
}
