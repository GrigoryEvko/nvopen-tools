// Function: sub_EE6E80
// Address: 0xee6e80
//
__int64 __fastcall sub_EE6E80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  unsigned __int64 *v7; // r15
  char v8; // si
  __int64 v9; // rsi
  unsigned __int64 *v10; // r12
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 i; // rax
  unsigned __int64 v14; // r10
  unsigned __int64 v15; // r10
  __int64 v16; // rax
  _QWORD **v17; // rsi
  _QWORD *v18; // rax
  __int64 v19; // r15
  __int64 *v20; // rax
  __int64 v21; // rax
  __int64 v23; // rax
  unsigned __int64 *v24; // rcx
  _QWORD *v25; // rdx
  __int64 *v26; // rdx
  char v28; // [rsp+17h] [rbp-D9h]
  int v29; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v30; // [rsp+18h] [rbp-D8h]
  __int64 *v31; // [rsp+28h] [rbp-C8h] BYREF
  _QWORD *v32; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v33; // [rsp+38h] [rbp-B8h]
  _QWORD v34[22]; // [rsp+40h] [rbp-B0h] BYREF

  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(unsigned __int64 **)a2;
  v8 = *(_BYTE *)(a1 + 129);
  v32 = v34;
  v33 = 0x2000000002LL;
  v28 = v8;
  v9 = v6;
  v10 = &v7[v6];
  v34[0] = 0;
  sub_D953B0((__int64)&v32, v9, a3, a4, a5, a6);
  for ( i = (unsigned int)v33; v10 != v7; LODWORD(v33) = v33 + 1 )
  {
    v14 = *v7;
    if ( i + 1 > (unsigned __int64)HIDWORD(v33) )
    {
      v30 = *v7;
      sub_C8D5F0((__int64)&v32, v34, i + 1, 4u, v11, v12);
      i = (unsigned int)v33;
      v14 = v30;
    }
    *((_DWORD *)v32 + i) = v14;
    v15 = HIDWORD(v14);
    LODWORD(v33) = v33 + 1;
    v16 = (unsigned int)v33;
    if ( (unsigned __int64)(unsigned int)v33 + 1 > HIDWORD(v33) )
    {
      v29 = v15;
      sub_C8D5F0((__int64)&v32, v34, (unsigned int)v33 + 1LL, 4u, v11, v12);
      v16 = (unsigned int)v33;
      LODWORD(v15) = v29;
    }
    ++v7;
    *((_DWORD *)v32 + v16) = v15;
    i = (unsigned int)(v33 + 1);
  }
  v17 = &v32;
  v18 = sub_C65B40(a1 + 96, (__int64)&v32, (__int64 *)&v31, (__int64)off_497B2F0);
  v19 = (__int64)v18;
  if ( v18 )
  {
    v19 = (__int64)(v18 + 1);
    if ( v32 != v34 )
      _libc_free(v32, &v32);
    v32 = (_QWORD *)v19;
    v20 = sub_EE6840(a1 + 136, (__int64 *)&v32);
    if ( v20 )
    {
      v21 = v20[1];
      if ( v21 )
        v19 = v21;
    }
    if ( *(_QWORD *)(a1 + 120) == v19 )
      *(_BYTE *)(a1 + 128) = 1;
  }
  else
  {
    if ( v28 )
    {
      v23 = sub_CD1D40((__int64 *)a1, 40, 3);
      *(_QWORD *)v23 = 0;
      v17 = (_QWORD **)v23;
      v19 = v23 + 8;
      v24 = *(unsigned __int64 **)a2;
      v25 = *(_QWORD **)(a2 + 8);
      *(_WORD *)(v23 + 16) = 0x4000;
      LOBYTE(v23) = *(_BYTE *)(v23 + 18);
      v17[3] = v24;
      v17[4] = v25;
      v26 = v31;
      *((_BYTE *)v17 + 18) = v23 & 0xF0 | 5;
      v17[1] = &unk_49DED68;
      sub_C657C0((__int64 *)(a1 + 96), (__int64 *)v17, v26, (__int64)off_497B2F0);
    }
    if ( v32 != v34 )
      _libc_free(v32, v17);
    *(_QWORD *)(a1 + 112) = v19;
  }
  return v19;
}
