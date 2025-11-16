// Function: sub_D33D80
// Address: 0xd33d80
//
__int64 __fastcall sub_D33D80(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // r9
  __int64 v7; // r13
  signed __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // rbx
  _BYTE *v12; // r8
  unsigned int v13; // ebx
  size_t v14; // r10
  _BYTE *v15; // rdi
  _BYTE *v16; // rdi
  _BYTE *src; // [rsp+8h] [rbp-A8h]
  __int64 v18; // [rsp+18h] [rbp-98h]
  _BYTE *v19; // [rsp+20h] [rbp-90h] BYREF
  __int64 v20; // [rsp+28h] [rbp-88h]
  _BYTE v21[32]; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v22; // [rsp+50h] [rbp-60h] BYREF
  __int64 v23; // [rsp+58h] [rbp-58h]
  _BYTE v24[80]; // [rsp+60h] [rbp-50h] BYREF

  v5 = a1[5];
  v6 = a1[4];
  if ( v5 == 2 )
    return *(_QWORD *)(v6 + 8);
  v9 = 8 * v5 - 8;
  v10 = a1[6];
  v19 = v21;
  v20 = 0x300000000LL;
  v11 = v9 >> 3;
  if ( (unsigned __int64)v9 > 0x18 )
  {
    v18 = v6;
    sub_C8D5F0((__int64)&v19, v21, (8 * v5 - 8) >> 3, 8u, a5, v6);
    v6 = v18;
    v15 = &v19[8 * (unsigned int)v20];
  }
  else
  {
    v12 = v21;
    if ( 8 * v5 == 8 )
      goto LABEL_6;
    v15 = v21;
  }
  memcpy(v15, (const void *)(v6 + 8), v9);
  v12 = v19;
  LODWORD(v9) = v20;
LABEL_6:
  v13 = v9 + v11;
  LODWORD(v20) = v13;
  v22 = v24;
  v14 = 8LL * v13;
  v23 = 0x400000000LL;
  if ( v13 > 4uLL )
  {
    src = v12;
    sub_C8D5F0((__int64)&v22, v24, v13, 8u, (__int64)v12, (__int64)&v22);
    v14 = 8LL * v13;
    v12 = src;
    v16 = &v22[8 * (unsigned int)v23];
LABEL_15:
    memcpy(v16, v12, v14);
    LODWORD(v14) = v23;
    goto LABEL_8;
  }
  if ( v14 )
  {
    v16 = v24;
    goto LABEL_15;
  }
LABEL_8:
  LODWORD(v23) = v14 + v13;
  v7 = sub_DBFF60(a2, &v22, v10, 0, v12);
  if ( v22 != v24 )
    _libc_free(v22, &v22);
  if ( v19 != v21 )
    _libc_free(v19, &v22);
  return v7;
}
