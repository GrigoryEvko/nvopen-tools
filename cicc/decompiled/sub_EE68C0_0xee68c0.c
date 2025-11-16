// Function: sub_EE68C0
// Address: 0xee68c0
//
__int64 __fastcall sub_EE68C0(__int64 a1, char *a2)
{
  char v3; // al
  size_t v4; // rax
  __int64 *v5; // rsi
  _QWORD *v6; // rax
  __int64 v7; // r15
  __int64 *v8; // rax
  __int64 v9; // rax
  __int64 *v11; // rax
  __int64 *v12; // r13
  size_t v13; // rax
  size_t v14; // r10
  __int64 *v15; // rdx
  char v16; // [rsp+7h] [rbp-D9h]
  __int64 *v17; // [rsp+18h] [rbp-C8h] BYREF
  __int64 v18[2]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD v19[22]; // [rsp+30h] [rbp-B0h] BYREF

  v3 = *(_BYTE *)(a1 + 129);
  v18[0] = (__int64)v19;
  v19[0] = 8;
  v16 = v3;
  v18[1] = 0x2000000002LL;
  v4 = strlen(a2);
  if ( v4 )
    sub_C653C0((__int64)v18, (unsigned __int8 *)a2, v4);
  else
    sub_C653C0((__int64)v18, 0, 0);
  v5 = v18;
  v6 = sub_C65B40(a1 + 96, (__int64)v18, (__int64 *)&v17, (__int64)off_497B2F0);
  v7 = (__int64)v6;
  if ( v6 )
  {
    v7 = (__int64)(v6 + 1);
    if ( (_QWORD *)v18[0] != v19 )
      _libc_free(v18[0], v18);
    v18[0] = v7;
    v8 = sub_EE6840(a1 + 136, v18);
    if ( v8 )
    {
      v9 = v8[1];
      if ( v9 )
        v7 = v9;
    }
    if ( *(_QWORD *)(a1 + 120) == v7 )
      *(_BYTE *)(a1 + 128) = 1;
  }
  else
  {
    if ( v16 )
    {
      v11 = (__int64 *)sub_CD1D40((__int64 *)a1, 40, 3);
      *v11 = 0;
      v12 = v11;
      v7 = (__int64)(v11 + 1);
      v13 = strlen(a2);
      v12[4] = (__int64)a2;
      v14 = v13;
      v15 = v17;
      v5 = v12;
      *((_WORD *)v12 + 8) = 16392;
      LOBYTE(v13) = *((_BYTE *)v12 + 18);
      v12[3] = v14;
      *((_BYTE *)v12 + 18) = v13 & 0xF0 | 5;
      v12[1] = (__int64)&unk_49DEFA8;
      sub_C657C0((__int64 *)(a1 + 96), v12, v15, (__int64)off_497B2F0);
    }
    if ( (_QWORD *)v18[0] != v19 )
      _libc_free(v18[0], v5);
    *(_QWORD *)(a1 + 112) = v7;
  }
  return v7;
}
