// Function: sub_2F78B80
// Address: 0x2f78b80
//
void __fastcall sub_2F78B80(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // rsi
  __int16 v5; // ax
  __int64 v6; // rdx
  __int64 v7; // r8
  __int64 v8; // rcx
  __int64 v9; // rcx
  __int64 v10; // r8
  unsigned __int64 v11; // rcx
  __int64 v12; // rsi
  unsigned __int64 i; // rax
  __int64 j; // rdi
  __int16 v15; // dx
  __int64 v16; // rdi
  __int64 v17; // rsi
  unsigned int v18; // ecx
  __int64 *v19; // rdx
  __int64 v20; // r8
  int v21; // edx
  int v22; // r10d
  __int64 v23[2]; // [rsp+10h] [rbp-2A0h] BYREF
  _BYTE v24[192]; // [rsp+20h] [rbp-290h] BYREF
  _BYTE *v25; // [rsp+E0h] [rbp-1D0h]
  __int64 v26; // [rsp+E8h] [rbp-1C8h]
  _BYTE v27[192]; // [rsp+F0h] [rbp-1C0h] BYREF
  _BYTE *v28; // [rsp+1B0h] [rbp-100h]
  __int64 v29; // [rsp+1B8h] [rbp-F8h]
  _BYTE v30[240]; // [rsp+1C0h] [rbp-F0h] BYREF

  sub_2F771D0(a1, a2);
  v4 = *(_QWORD *)(a1 + 64);
  v5 = *(_WORD *)(v4 + 68);
  if ( (unsigned __int16)(v5 - 14) <= 4u || v5 == 24 )
    return;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = *(unsigned __int8 *)(a1 + 58);
  v25 = v27;
  v8 = *(_QWORD *)(a1 + 24);
  v23[0] = (__int64)v24;
  v23[1] = 0x800000000LL;
  v26 = 0x800000000LL;
  v28 = v30;
  v29 = 0x800000000LL;
  sub_2F75980((__int64)v23, v4, v6, v8, v7, 0);
  if ( !*(_BYTE *)(a1 + 58) )
  {
    if ( *(_BYTE *)(a1 + 56) )
      sub_2F761E0((__int64)v23, v4, *(_QWORD *)(a1 + 32), v9, v10);
    goto LABEL_7;
  }
  v11 = *(_QWORD *)(a1 + 64);
  v12 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 32LL);
  for ( i = v11; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  for ( ; (*(_BYTE *)(v11 + 44) & 8) != 0; v11 = *(_QWORD *)(v11 + 8) )
    ;
  for ( j = *(_QWORD *)(v11 + 8); j != i; i = *(_QWORD *)(i + 8) )
  {
    v15 = *(_WORD *)(i + 68);
    if ( (unsigned __int16)(v15 - 14) > 4u && v15 != 24 )
      break;
  }
  v16 = *(_QWORD *)(v12 + 128);
  v17 = *(unsigned int *)(v12 + 144);
  if ( !(_DWORD)v17 )
    goto LABEL_24;
  v18 = (v17 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
  v19 = (__int64 *)(v16 + 16LL * v18);
  v20 = *v19;
  if ( i != *v19 )
  {
    v21 = 1;
    while ( v20 != -4096 )
    {
      v22 = v21 + 1;
      v18 = (v17 - 1) & (v21 + v18);
      v19 = (__int64 *)(v16 + 16LL * v18);
      v20 = *v19;
      if ( i == *v19 )
        goto LABEL_23;
      v21 = v22;
    }
LABEL_24:
    v19 = (__int64 *)(v16 + 16 * v17);
  }
LABEL_23:
  sub_2F76630(v23, *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 24), v19[1] & 0xFFFFFFFFFFFFFFF8LL | 4, 0);
LABEL_7:
  sub_2F78160(a1, v23, a2);
  if ( v28 != v30 )
    _libc_free((unsigned __int64)v28);
  if ( v25 != v27 )
    _libc_free((unsigned __int64)v25);
  if ( (_BYTE *)v23[0] != v24 )
    _libc_free(v23[0]);
}
