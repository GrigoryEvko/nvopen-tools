// Function: sub_EE8100
// Address: 0xee8100
//
__int64 __fastcall sub_EE8100(__int64 a1, __int64 *a2, int *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v8; // r14
  int v9; // r12d
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 *v15; // rsi
  _QWORD *v16; // rax
  __int64 v17; // r12
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v21; // rax
  __int64 v22; // rcx
  __int16 v23; // dx
  int v24; // edi
  __int64 *v25; // rdx
  char v27; // [rsp+17h] [rbp-D9h]
  __int64 *v28; // [rsp+18h] [rbp-D8h]
  __int64 *v29; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v30[2]; // [rsp+30h] [rbp-C0h] BYREF
  _QWORD v31[22]; // [rsp+40h] [rbp-B0h] BYREF

  v8 = (__int64 *)(a1 + 96);
  v9 = *a3;
  v10 = *a2;
  v28 = a2;
  v27 = *(_BYTE *)(a1 + 129);
  v30[1] = 0x2000000002LL;
  v30[0] = (__int64)v31;
  v31[0] = 13;
  sub_D953B0((__int64)v30, v10, (__int64)a3, a4, (__int64)a2, a6);
  sub_D953B0((__int64)v30, v9, v11, v12, v13, v14);
  v15 = v30;
  v16 = sub_C65B40(a1 + 96, (__int64)v30, (__int64 *)&v29, (__int64)off_497B2F0);
  v17 = (__int64)v16;
  if ( v16 )
  {
    v17 = (__int64)(v16 + 1);
    if ( (_QWORD *)v30[0] != v31 )
      _libc_free(v30[0], v30);
    v30[0] = v17;
    v18 = sub_EE6840(a1 + 136, v30);
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
    if ( v27 )
    {
      v21 = sub_CD1D40((__int64 *)a1, 40, 3);
      *(_QWORD *)v21 = 0;
      v15 = (__int64 *)v21;
      v22 = *v28;
      v23 = *(_WORD *)(v21 + 16);
      v17 = v21 + 8;
      v24 = *a3;
      *(_BYTE *)(v21 + 36) = 0;
      LOBYTE(v21) = *(_BYTE *)(v22 + 9);
      v15[3] = v22;
      *((_DWORD *)v15 + 8) = v24;
      *((_WORD *)v15 + 8) = v23 & 0xC000 | 0xD;
      LOWORD(v21) = (unsigned __int8)v21 >> 6 << 6;
      BYTE1(v21) |= 5u;
      v25 = v29;
      *(_WORD *)((char *)v15 + 17) = *(_WORD *)((_BYTE *)v15 + 17) & 0xF03F | v21;
      v15[1] = (__int64)&unk_49DF2A8;
      sub_C657C0(v8, v15, v25, (__int64)off_497B2F0);
    }
    if ( (_QWORD *)v30[0] != v31 )
      _libc_free(v30[0], v15);
    *(_QWORD *)(a1 + 112) = v17;
  }
  return v17;
}
