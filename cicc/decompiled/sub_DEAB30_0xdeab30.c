// Function: sub_DEAB30
// Address: 0xdeab30
//
__int64 __fastcall sub_DEAB30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // rax
  __int64 v8; // r9
  __int64 v9; // r12
  void *v10; // r8
  int v11; // r14d
  size_t v12; // r15
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  void *v16; // [rsp+8h] [rbp-E8h]
  void *src; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v18; // [rsp+18h] [rbp-D8h]
  _BYTE v19[48]; // [rsp+20h] [rbp-D0h] BYREF
  _QWORD v20[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v21; // [rsp+60h] [rbp-90h]
  __int64 v22; // [rsp+68h] [rbp-88h] BYREF
  unsigned int v23; // [rsp+70h] [rbp-80h]
  _QWORD v24[9]; // [rsp+A8h] [rbp-48h] BYREF

  src = v19;
  v20[0] = a1;
  v20[1] = 0;
  v21 = 1;
  v18 = 0x600000000LL;
  v7 = &v22;
  do
  {
    *v7 = -4096;
    v7 += 2;
  }
  while ( v7 != v24 );
  v24[2] = a3;
  v24[0] = &src;
  v24[1] = 0;
  v9 = sub_DE9D10(v20, a2, (__int64)v24, a4, a1, a6);
  if ( (v21 & 1) == 0 )
  {
    a2 = 16LL * v23;
    sub_C7D6A0(v22, a2, 8);
  }
  if ( *(_WORD *)(v9 + 24) == 8 )
  {
    v10 = src;
    v11 = v18;
    v12 = 8LL * (unsigned int)v18;
    v13 = *(unsigned int *)(a4 + 8);
    v14 = (unsigned int)v18 + v13;
    if ( v14 > *(unsigned int *)(a4 + 12) )
    {
      a2 = a4 + 16;
      v16 = src;
      sub_C8D5F0(a4, (const void *)(a4 + 16), v14, 8u, (__int64)src, v8);
      v13 = *(unsigned int *)(a4 + 8);
      v10 = v16;
      if ( !v12 )
        goto LABEL_9;
    }
    else if ( !v12 )
    {
LABEL_9:
      *(_DWORD *)(a4 + 8) = v11 + v13;
      goto LABEL_10;
    }
    a2 = (__int64)v10;
    memcpy((void *)(*(_QWORD *)a4 + 8 * v13), v10, v12);
    LODWORD(v13) = *(_DWORD *)(a4 + 8);
    goto LABEL_9;
  }
  v9 = 0;
LABEL_10:
  if ( src != v19 )
    _libc_free(src, a2);
  return v9;
}
