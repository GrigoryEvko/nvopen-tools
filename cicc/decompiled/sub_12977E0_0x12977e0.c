// Function: sub_12977E0
// Address: 0x12977e0
//
__int64 __fastcall sub_12977E0(_QWORD **a1, __int64 a2, __int64 *a3, char a4)
{
  __int64 *v5; // r13
  int v6; // r8d
  __int64 v7; // rax
  __int64 v8; // rax
  bool v9; // r8
  __int64 v10; // rax
  char v11; // r8
  __int64 v12; // rax
  char v13; // r8
  __int64 v14; // rdi
  __int64 v15; // r12
  char v17; // [rsp+Fh] [rbp-161h]
  char v18; // [rsp+Fh] [rbp-161h]
  bool v19; // [rsp+Fh] [rbp-161h]
  _BYTE *v21; // [rsp+50h] [rbp-120h] BYREF
  __int64 v22; // [rsp+58h] [rbp-118h]
  _BYTE v23[16]; // [rsp+60h] [rbp-110h] BYREF
  _BYTE *v24; // [rsp+70h] [rbp-100h] BYREF
  __int64 v25; // [rsp+78h] [rbp-F8h]
  _BYTE v26[16]; // [rsp+80h] [rbp-F0h] BYREF
  _BYTE *v27; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v28; // [rsp+98h] [rbp-D8h]
  _BYTE v29[16]; // [rsp+A0h] [rbp-D0h] BYREF
  _BYTE *v30; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v31; // [rsp+B8h] [rbp-B8h]
  _BYTE v32[176]; // [rsp+C0h] [rbp-B0h] BYREF

  v30 = v32;
  v21 = v23;
  v31 = 0x1000000000LL;
  v22 = 0x1000000000LL;
  v24 = v26;
  v25 = 0x1000000000LL;
  v27 = v29;
  v28 = 0x1000000000LL;
  if ( a3 )
  {
    v5 = a3;
    while ( !dword_4F0690C )
    {
      v14 = v5[1];
      LOBYTE(v6) = 0;
      if ( (*(_BYTE *)(v14 + 140) & 0xFB) != 8 )
        goto LABEL_4;
      LOBYTE(v6) = (unsigned int)sub_8D4C10(v14, dword_4F077C4 != 2) >> 2;
      v7 = (unsigned int)v22;
      LOBYTE(v6) = v6 & 1;
      if ( (unsigned int)v22 >= HIDWORD(v22) )
      {
LABEL_17:
        v17 = v6;
        sub_16CD150(&v21, v23, 0, 1);
        v7 = (unsigned int)v22;
        LOBYTE(v6) = v17;
      }
LABEL_5:
      v21[v7] = v6;
      v8 = (unsigned int)v31;
      LODWORD(v22) = v22 + 1;
      if ( (unsigned int)v31 >= HIDWORD(v31) )
      {
        sub_16CD150(&v30, v32, 0, 8);
        v8 = (unsigned int)v31;
      }
      v9 = 0;
      *(_QWORD *)&v30[8 * v8] = v5[1];
      LODWORD(v31) = v31 + 1;
      if ( a4 )
        v9 = (v5[4] & 2) != 0;
      v10 = (unsigned int)v25;
      if ( (unsigned int)v25 >= HIDWORD(v25) )
      {
        v19 = v9;
        sub_16CD150(&v24, v26, 0, 1);
        v10 = (unsigned int)v25;
        v9 = v19;
      }
      v24[v10] = v9;
      v11 = *((_BYTE *)v5 + 32);
      LODWORD(v25) = v25 + 1;
      v12 = (unsigned int)v28;
      v13 = v11 & 1;
      if ( (unsigned int)v28 >= HIDWORD(v28) )
      {
        v18 = v13;
        sub_16CD150(&v27, v29, 0, 1);
        v12 = (unsigned int)v28;
        v13 = v18;
      }
      v27[v12] = v13;
      LODWORD(v28) = v28 + 1;
      v5 = (__int64 *)*v5;
      if ( !v5 )
        goto LABEL_18;
    }
    v6 = (*((_DWORD *)v5 + 8) >> 13) & 1;
LABEL_4:
    v7 = (unsigned int)v22;
    if ( (unsigned int)v22 >= HIDWORD(v22) )
      goto LABEL_17;
    goto LABEL_5;
  }
LABEL_18:
  v15 = sub_12975B0(a1, a2, (__int64)&v30, (__int64 *)&v21, (__int64 *)&v24, (__int64 *)&v27);
  if ( v27 != v29 )
    _libc_free(v27, a2);
  if ( v24 != v26 )
    _libc_free(v24, a2);
  if ( v21 != v23 )
    _libc_free(v21, a2);
  if ( v30 != v32 )
    _libc_free(v30, a2);
  return v15;
}
