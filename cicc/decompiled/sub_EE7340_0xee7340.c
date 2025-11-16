// Function: sub_EE7340
// Address: 0xee7340
//
__int64 __fastcall sub_EE7340(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rsi
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r8
  __int64 v15; // rax
  unsigned __int64 *v16; // r13
  unsigned __int64 *i; // r15
  unsigned __int64 v18; // r9
  __int64 v19; // r9
  __int64 v20; // rax
  _QWORD **v21; // rsi
  _QWORD *v22; // rax
  __int64 v23; // r15
  __int64 *v24; // rax
  __int64 v25; // rax
  __int64 v27; // rax
  _QWORD *v28; // rcx
  _QWORD *v29; // rdx
  __int64 *v30; // rdx
  char v33; // [rsp+17h] [rbp-D9h]
  unsigned __int64 *v34; // [rsp+18h] [rbp-D8h]
  int v35; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v36; // [rsp+18h] [rbp-D8h]
  __int64 *v37; // [rsp+28h] [rbp-C8h] BYREF
  _QWORD *v38; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v39; // [rsp+38h] [rbp-B8h]
  _QWORD v40[22]; // [rsp+40h] [rbp-B0h] BYREF

  v6 = *a2;
  v33 = *(_BYTE *)(a1 + 129);
  v8 = *a3;
  v9 = a3[1];
  v38 = v40;
  v34 = (unsigned __int64 *)v8;
  v39 = 0x2000000002LL;
  v40[0] = 70;
  sub_D953B0((__int64)&v38, v6, v8, a4, a5, a6);
  sub_D953B0((__int64)&v38, v9, v10, v11, v12, v13);
  v15 = (unsigned int)v39;
  v16 = &v34[v9];
  for ( i = v34; v16 != i; LODWORD(v39) = v39 + 1 )
  {
    v18 = *i;
    if ( v15 + 1 > (unsigned __int64)HIDWORD(v39) )
    {
      v36 = *i;
      sub_C8D5F0((__int64)&v38, v40, v15 + 1, 4u, v14, v18);
      v15 = (unsigned int)v39;
      v18 = v36;
    }
    *((_DWORD *)v38 + v15) = v18;
    v19 = HIDWORD(v18);
    LODWORD(v39) = v39 + 1;
    v20 = (unsigned int)v39;
    if ( (unsigned __int64)(unsigned int)v39 + 1 > HIDWORD(v39) )
    {
      v35 = v19;
      sub_C8D5F0((__int64)&v38, v40, (unsigned int)v39 + 1LL, 4u, v14, v19);
      v20 = (unsigned int)v39;
      LODWORD(v19) = v35;
    }
    ++i;
    *((_DWORD *)v38 + v20) = v19;
    v15 = (unsigned int)(v39 + 1);
  }
  v21 = &v38;
  v22 = sub_C65B40(a1 + 96, (__int64)&v38, (__int64 *)&v37, (__int64)off_497B2F0);
  v23 = (__int64)v22;
  if ( v22 )
  {
    v23 = (__int64)(v22 + 1);
    if ( v38 != v40 )
      _libc_free(v38, &v38);
    v38 = (_QWORD *)v23;
    v24 = sub_EE6840(a1 + 136, (__int64 *)&v38);
    if ( v24 )
    {
      v25 = v24[1];
      if ( v25 )
        v23 = v25;
    }
    if ( *(_QWORD *)(a1 + 120) == v23 )
      *(_BYTE *)(a1 + 128) = 1;
  }
  else
  {
    if ( v33 )
    {
      v27 = sub_CD1D40((__int64 *)a1, 48, 3);
      *(_QWORD *)v27 = 0;
      v21 = (_QWORD **)v27;
      v23 = v27 + 8;
      v28 = (_QWORD *)*a3;
      v29 = (_QWORD *)a3[1];
      *(_QWORD *)(v27 + 24) = *a2;
      *(_WORD *)(v27 + 16) = 16454;
      LOBYTE(v27) = *(_BYTE *)(v27 + 18);
      v21[4] = v28;
      v21[5] = v29;
      v30 = v37;
      *((_BYTE *)v21 + 18) = v27 & 0xF0 | 5;
      v21[1] = &unk_49E0808;
      sub_C657C0((__int64 *)(a1 + 96), (__int64 *)v21, v30, (__int64)off_497B2F0);
    }
    if ( v38 != v40 )
      _libc_free(v38, v21);
    *(_QWORD *)(a1 + 112) = v23;
  }
  return v23;
}
