// Function: sub_EE28B0
// Address: 0xee28b0
//
__int64 __fastcall sub_EE28B0(__int64 a1)
{
  __int64 v1; // r15
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned __int64 v7; // rbx
  __int64 v8; // rax
  _BYTE **v9; // rsi
  _BYTE **v10; // rsi
  unsigned __int64 *(__fastcall *v11)(unsigned __int64 *, __int64, __int64, __int64, __int64, __int64); // rax
  _BYTE *v12; // rax
  __int64 v13; // rdx
  size_t *v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned __int64 v17; // r8
  unsigned __int64 v18; // r9
  unsigned __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rdi
  int v22; // [rsp+24h] [rbp-BCh] BYREF
  __int64 v23; // [rsp+28h] [rbp-B8h] BYREF
  __int64 v24; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v25; // [rsp+38h] [rbp-A8h] BYREF
  unsigned __int64 v26; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v27; // [rsp+48h] [rbp-98h] BYREF
  _BYTE *v28; // [rsp+50h] [rbp-90h] BYREF
  __int64 v29; // [rsp+58h] [rbp-88h]
  _QWORD v30[2]; // [rsp+60h] [rbp-80h] BYREF
  size_t *v31; // [rsp+70h] [rbp-70h] BYREF
  __int64 v32[2]; // [rsp+78h] [rbp-68h] BYREF
  _QWORD v33[11]; // [rsp+88h] [rbp-58h] BYREF

  v1 = *(_QWORD *)(a1 + 48);
  if ( !v1 )
  {
    v4 = sub_22077B0(400);
    v1 = v4;
    if ( v4 )
    {
      *(_QWORD *)v4 = 0;
      *(_QWORD *)(v4 + 8) = 0;
      *(_QWORD *)(v4 + 16) = 0;
      *(_QWORD *)(v4 + 24) = 0;
      *(_QWORD *)(v4 + 32) = 0;
      *(_QWORD *)(v4 + 40) = 0x800000000LL;
      *(_QWORD *)(v4 + 64) = 0x800000000LL;
      *(_QWORD *)(v4 + 200) = v4 + 216;
      *(_QWORD *)(v4 + 208) = 0x400000000LL;
      *(_QWORD *)(v4 + 248) = v4 + 264;
      *(_QWORD *)(v4 + 384) = v4 + 176;
      *(_QWORD *)(v4 + 48) = 0;
      *(_QWORD *)(v4 + 56) = 0;
      *(_QWORD *)(v4 + 72) = 0;
      *(_QWORD *)(v4 + 80) = 0;
      *(_QWORD *)(v4 + 88) = 0;
      *(_QWORD *)(v4 + 96) = 0;
      *(_QWORD *)(v4 + 104) = 0;
      *(_QWORD *)(v4 + 112) = 0;
      *(_QWORD *)(v4 + 120) = 0;
      *(_QWORD *)(v4 + 128) = 0;
      *(_QWORD *)(v4 + 136) = 0;
      *(_DWORD *)(v4 + 144) = 0;
      *(_QWORD *)(v4 + 152) = 0;
      *(_QWORD *)(v4 + 160) = 0;
      *(_QWORD *)(v4 + 168) = 0;
      *(_QWORD *)(v4 + 176) = 0;
      *(_QWORD *)(v4 + 184) = 0;
      *(_QWORD *)(v4 + 192) = 0;
      *(_QWORD *)(v4 + 256) = 0;
      *(_QWORD *)(v4 + 264) = 0;
      *(_QWORD *)(v4 + 272) = 1;
      memset((void *)(v4 + 280), 0, 0x60u);
      *(_BYTE *)(v4 + 392) = 0;
      *(_QWORD *)(v4 + 376) = 0;
    }
    sub_ED3B20(&v23, v4, *(char **)(a1 + 456), *(_QWORD *)(a1 + 464));
    v7 = v23 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v23 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v23 = 0;
      v22 = 0;
      sub_ED6550((__int64 *)&v28, byte_3F871B3);
      v26 = v7 | 1;
      v31 = (size_t *)&v22;
      v32[0] = (__int64)&v28;
      v24 = 0;
      v25 = 0;
      sub_EDA620(&v27, &v26, (__int64)&v31);
      if ( (v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        BUG();
      v27 = 0;
      sub_9C66B0(&v27);
      sub_9C66B0((__int64 *)&v26);
      sub_9C66B0(&v25);
      LODWORD(v31) = v22;
      v32[0] = (__int64)v33;
      sub_ED71E0(v32, v28, (__int64)&v28[v29]);
      if ( v28 != (_BYTE *)v30 )
        j_j___libc_free_0(v28, v30[0] + 1LL);
      sub_9C66B0(&v24);
      sub_ED85B0(&v25, a1, (int)v31, v32);
      v8 = v25;
      v9 = &v28;
      v25 = 0;
      v26 = 0;
      v28 = (_BYTE *)(v8 | 1);
      sub_EDA3C0(&v27, (__int64 *)&v28);
      if ( (v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        BUG();
      v27 = 0;
      sub_9C66B0(&v27);
      sub_9C66B0((__int64 *)&v28);
      sub_9C66B0((__int64 *)&v26);
      sub_9C66B0(&v25);
      if ( (_QWORD *)v32[0] != v33 )
      {
        v9 = (_BYTE **)(v33[0] + 1LL);
        j_j___libc_free_0(v32[0], v33[0] + 1LL);
      }
      if ( (v23 & 1) != 0 || (v23 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v23, (__int64)v9);
    }
    v10 = *(_BYTE ***)(a1 + 128);
    v11 = (unsigned __int64 *(__fastcall *)(unsigned __int64 *, __int64, __int64, __int64, __int64, __int64))*((_QWORD *)*v10 + 17);
    if ( v11 == sub_EE2840 )
    {
      v12 = v10[1];
      v10 = (_BYTE **)v1;
      v13 = *((_QWORD *)v12 + 1);
      v14 = (size_t *)*((_QWORD *)v12 + 9);
      v32[0] = 0;
      v33[0] = v12 + 32;
      v32[1] = v13;
      v31 = v14;
      memset(&v33[1], 0, 32);
      sub_EE2740((unsigned __int64 *)&v23, v1, &v31, (__int64)v14, v5, v6);
    }
    else
    {
      ((void (__fastcall *)(__int64 *, _BYTE **, __int64))v11)(&v23, v10, v1);
    }
    v19 = v23 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v23 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v23 = 0;
      v22 = 0;
      sub_ED6550((__int64 *)&v28, byte_3F871B3);
      v26 = v19 | 1;
      v31 = (size_t *)&v22;
      v32[0] = (__int64)&v28;
      v24 = 0;
      v25 = 0;
      sub_EDA620(&v27, &v26, (__int64)&v31);
      if ( (v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        BUG();
      v27 = 0;
      sub_9C66B0(&v27);
      sub_9C66B0((__int64 *)&v26);
      sub_9C66B0(&v25);
      LODWORD(v31) = v22;
      v32[0] = (__int64)v33;
      sub_ED71E0(v32, v28, (__int64)&v28[v29]);
      if ( v28 != (_BYTE *)v30 )
        j_j___libc_free_0(v28, v30[0] + 1LL);
      sub_9C66B0(&v24);
      sub_ED85B0(&v25, a1, (int)v31, v32);
      v20 = v25;
      v10 = &v28;
      v25 = 0;
      v26 = 0;
      v28 = (_BYTE *)(v20 | 1);
      sub_EDA3C0(&v27, (__int64 *)&v28);
      if ( (v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        BUG();
      v27 = 0;
      sub_9C66B0(&v27);
      sub_9C66B0((__int64 *)&v28);
      sub_9C66B0((__int64 *)&v26);
      sub_9C66B0(&v25);
      if ( (_QWORD *)v32[0] != v33 )
      {
        v10 = (_BYTE **)(v33[0] + 1LL);
        j_j___libc_free_0(v32[0], v33[0] + 1LL);
      }
      if ( (v23 & 1) != 0 || (v23 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_C63C30(&v23, (__int64)v10);
    }
    v21 = *(_QWORD *)(a1 + 48);
    *(_QWORD *)(a1 + 48) = v1;
    if ( v21 )
    {
      sub_EDAD10(v21, (__int64)v10, v15, v16, v17, v18);
      return *(_QWORD *)(a1 + 48);
    }
  }
  return v1;
}
