// Function: sub_1EE8D00
// Address: 0x1ee8d00
//
void __fastcall sub_1EE8D00(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rsi
  __int64 v4; // rcx
  __int64 v5; // rdx
  char v6; // r8
  unsigned __int64 v7; // rax
  __int64 i; // rcx
  __int64 v9; // rdi
  __int64 v10; // rcx
  unsigned int v11; // esi
  __int64 *v12; // rdx
  __int64 v13; // r8
  int v14; // edx
  int v15; // r10d
  unsigned __int64 v17[2]; // [rsp+10h] [rbp-120h] BYREF
  _BYTE v18[64]; // [rsp+20h] [rbp-110h] BYREF
  _BYTE *v19; // [rsp+60h] [rbp-D0h]
  __int64 v20; // [rsp+68h] [rbp-C8h]
  _BYTE v21[64]; // [rsp+70h] [rbp-C0h] BYREF
  _BYTE *v22; // [rsp+B0h] [rbp-80h]
  __int64 v23; // [rsp+B8h] [rbp-78h]
  _BYTE v24[112]; // [rsp+C0h] [rbp-70h] BYREF

  sub_1EE76C0(a1, a2);
  v3 = *(_QWORD *)(a1 + 64);
  v4 = *(_QWORD *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v17[0] = (unsigned __int64)v18;
  v6 = *(_BYTE *)(a1 + 58);
  v17[1] = 0x800000000LL;
  v19 = v21;
  v20 = 0x800000000LL;
  v22 = v24;
  v23 = 0x800000000LL;
  sub_1EE65F0((__int64)v17, v3, v5, v4, v6, 0);
  if ( *(_BYTE *)(a1 + 58) )
  {
    v7 = *(_QWORD *)(a1 + 64);
    for ( i = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 272LL);
          (*(_BYTE *)(v7 + 46) & 4) != 0;
          v7 = *(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL )
    {
      ;
    }
    v9 = *(_QWORD *)(i + 368);
    v10 = *(unsigned int *)(i + 384);
    if ( (_DWORD)v10 )
    {
      v11 = (v10 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v12 = (__int64 *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( v7 == *v12 )
      {
LABEL_15:
        sub_1EE6D60((__int64)v17, *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 24), v12[1] & 0xFFFFFFFFFFFFFFF8LL | 4, 0);
        goto LABEL_4;
      }
      v14 = 1;
      while ( v13 != -8 )
      {
        v15 = v14 + 1;
        v11 = (v10 - 1) & (v14 + v11);
        v12 = (__int64 *)(v9 + 16LL * v11);
        v13 = *v12;
        if ( v7 == *v12 )
          goto LABEL_15;
        v14 = v15;
      }
    }
    v12 = (__int64 *)(v9 + 16 * v10);
    goto LABEL_15;
  }
  if ( *(_BYTE *)(a1 + 56) )
    sub_1EE69C0((__int64)v17, v3, *(_QWORD *)(a1 + 32));
LABEL_4:
  sub_1EE8590(a1, (__int64)v17, a2);
  if ( v22 != v24 )
    _libc_free((unsigned __int64)v22);
  if ( v19 != v21 )
    _libc_free((unsigned __int64)v19);
  if ( (_BYTE *)v17[0] != v18 )
    _libc_free(v17[0]);
}
