// Function: sub_13A5BC0
// Address: 0x13a5bc0
//
__int64 __fastcall sub_13A5BC0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v6; // rax
  __int64 v7; // r13
  size_t v8; // r14
  const void *v9; // r10
  const void *v10; // r9
  __int64 v11; // rbx
  _BYTE *v12; // rdi
  int v13; // eax
  _BYTE *v14; // r8
  unsigned int v15; // ebx
  size_t v16; // r10
  _BYTE *v17; // rdi
  _BYTE *src; // [rsp+8h] [rbp-A8h]
  size_t n; // [rsp+10h] [rbp-A0h]
  const void *v20; // [rsp+18h] [rbp-98h]
  _BYTE *v21; // [rsp+20h] [rbp-90h] BYREF
  __int64 v22; // [rsp+28h] [rbp-88h]
  _BYTE dest[32]; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v24; // [rsp+50h] [rbp-60h] BYREF
  __int64 v25; // [rsp+58h] [rbp-58h]
  _BYTE v26[80]; // [rsp+60h] [rbp-50h] BYREF

  v2 = a1[5];
  v3 = a1[4];
  if ( v2 == 2 )
    return *(_QWORD *)(v3 + 8);
  v6 = 8 * v2;
  v7 = a1[6];
  v8 = v6 - 8;
  v21 = dest;
  v9 = (const void *)(v3 + v6);
  v22 = 0x300000000LL;
  v10 = (const void *)(v3 + 8);
  v11 = (v6 - 8) >> 3;
  if ( (unsigned __int64)(v6 - 8) > 0x18 )
  {
    n = v3 + v6;
    v20 = (const void *)(v3 + 8);
    sub_16CD150(&v21, dest, (v6 - 8) >> 3, 8);
    v14 = v21;
    v13 = v22;
    v10 = v20;
    v9 = (const void *)n;
    v12 = &v21[8 * (unsigned int)v22];
  }
  else
  {
    v12 = dest;
    v13 = 0;
    v14 = dest;
  }
  if ( v10 != v9 )
  {
    memcpy(v12, v10, v8);
    v14 = v21;
    v13 = v22;
  }
  v15 = v13 + v11;
  LODWORD(v22) = v15;
  v24 = v26;
  v16 = 8LL * v15;
  v25 = 0x400000000LL;
  if ( v15 > 4uLL )
  {
    src = v14;
    sub_16CD150(&v24, v26, v15, 8);
    v16 = 8LL * v15;
    v14 = src;
    v17 = &v24[8 * (unsigned int)v25];
LABEL_16:
    memcpy(v17, v14, v16);
    LODWORD(v16) = v25;
    goto LABEL_10;
  }
  if ( v16 )
  {
    v17 = v26;
    goto LABEL_16;
  }
LABEL_10:
  LODWORD(v25) = v16 + v15;
  v4 = sub_14785F0(a2, &v24, v7, 0, v14);
  if ( v24 != v26 )
    _libc_free((unsigned __int64)v24);
  if ( v21 != dest )
    _libc_free((unsigned __int64)v21);
  return v4;
}
