// Function: sub_EE7CC0
// Address: 0xee7cc0
//
__int64 __fastcall sub_EE7CC0(__int64 a1, __int64 *a2, unsigned __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  unsigned __int64 v8; // r15
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rax
  _QWORD **v15; // rsi
  _QWORD *v16; // rax
  __int64 v17; // r15
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v21; // rax
  _QWORD *v22; // rdx
  __int64 *v23; // rdx
  char v25; // [rsp+17h] [rbp-D9h]
  __int64 *v26; // [rsp+28h] [rbp-C8h] BYREF
  _QWORD *v27; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v28; // [rsp+38h] [rbp-B8h]
  _QWORD v29[22]; // [rsp+40h] [rbp-B0h] BYREF

  v7 = *a2;
  v8 = *a3;
  v25 = *(_BYTE *)(a1 + 129);
  v28 = 0x2000000002LL;
  v27 = v29;
  v29[0] = 45;
  sub_D953B0((__int64)&v27, v7, (__int64)a3, a4, a5, a6);
  v11 = (unsigned int)v28;
  v12 = (unsigned int)v28 + 1LL;
  if ( v12 > HIDWORD(v28) )
  {
    sub_C8D5F0((__int64)&v27, v29, v12, 4u, v9, v10);
    v11 = (unsigned int)v28;
  }
  v13 = HIDWORD(v8);
  *((_DWORD *)v27 + v11) = v8;
  LODWORD(v28) = v28 + 1;
  v14 = (unsigned int)v28;
  if ( (unsigned __int64)(unsigned int)v28 + 1 > HIDWORD(v28) )
  {
    sub_C8D5F0((__int64)&v27, v29, (unsigned int)v28 + 1LL, 4u, v13, v10);
    v14 = (unsigned int)v28;
    v13 = HIDWORD(v8);
  }
  v15 = &v27;
  *((_DWORD *)v27 + v14) = v13;
  LODWORD(v28) = v28 + 1;
  v16 = sub_C65B40(a1 + 96, (__int64)&v27, (__int64 *)&v26, (__int64)off_497B2F0);
  v17 = (__int64)v16;
  if ( v16 )
  {
    v17 = (__int64)(v16 + 1);
    if ( v27 != v29 )
      _libc_free(v27, &v27);
    v27 = (_QWORD *)v17;
    v18 = sub_EE6840(a1 + 136, (__int64 *)&v27);
    if ( v18 )
    {
      v19 = v18[1];
      if ( v19 )
        v17 = v19;
    }
    if ( *(_QWORD *)(a1 + 120) == v17 )
      *(_BYTE *)(a1 + 128) = 1;
  }
  else
  {
    if ( v25 )
    {
      v21 = sub_CD1D40((__int64 *)a1, 40, 3);
      *(_QWORD *)v21 = 0;
      v15 = (_QWORD **)v21;
      v17 = v21 + 8;
      v22 = (_QWORD *)*a3;
      *(_QWORD *)(v21 + 24) = *a2;
      *(_WORD *)(v21 + 16) = 16429;
      LOBYTE(v21) = *(_BYTE *)(v21 + 18);
      v15[4] = v22;
      v23 = v26;
      *((_BYTE *)v15 + 18) = v21 & 0xF0 | 5;
      v15[1] = &unk_49DFEA8;
      sub_C657C0((__int64 *)(a1 + 96), (__int64 *)v15, v23, (__int64)off_497B2F0);
    }
    if ( v27 != v29 )
      _libc_free(v27, v15);
    *(_QWORD *)(a1 + 112) = v17;
  }
  return v17;
}
