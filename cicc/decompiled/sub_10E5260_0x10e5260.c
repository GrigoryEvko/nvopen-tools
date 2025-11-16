// Function: sub_10E5260
// Address: 0x10e5260
//
unsigned __int8 *__fastcall sub_10E5260(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  _BYTE *v7; // rdi
  __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned __int64 v11; // rax
  unsigned __int8 v12; // cl
  __int64 *v13; // rbx
  __int64 v14; // r14
  _QWORD *v15; // rax
  __int64 v16; // r9
  __int64 v17; // r15
  __int64 v18; // r14
  __int64 v19; // rbx
  __int64 v20; // rdx
  unsigned int v21; // esi
  unsigned int **v22; // rdi
  int v23; // eax
  __int64 v24; // rax
  __int64 v26; // [rsp+0h] [rbp-B0h]
  unsigned int v27; // [rsp+Ch] [rbp-A4h]
  __int64 v28; // [rsp+18h] [rbp-98h]
  const char *v29; // [rsp+20h] [rbp-90h] BYREF
  char v30; // [rsp+40h] [rbp-70h]
  char v31; // [rsp+41h] [rbp-6Fh]
  __int64 v32[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v33; // [rsp+70h] [rbp-40h]

  v6 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v7 = *(_BYTE **)(a2 + 32 * (2 - v6));
  if ( *v7 > 0x15u )
    return 0;
  if ( !sub_AD7930(v7, a2, v6, a4, a5) )
    return 0;
  v9 = sub_9B7920(*(char **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( !v9 )
    return 0;
  v28 = *(_QWORD *)(a2 + 8);
  v10 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v11 = *(_QWORD *)(v10 + 24);
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
    v11 = *(_QWORD *)v11;
  v12 = 0;
  if ( v11 )
  {
    _BitScanReverse64(&v11, v11);
    v12 = 63 - (v11 ^ 0x3F);
  }
  v26 = v9;
  v13 = *(__int64 **)(a1 + 32);
  v14 = *(_QWORD *)(v28 + 24);
  v31 = 1;
  v29 = "load.scalar";
  v27 = v12;
  v30 = 3;
  v33 = 257;
  v15 = sub_BD2C40(80, unk_3F10A14);
  v16 = v27;
  v17 = (__int64)v15;
  if ( v15 )
    sub_B4D190((__int64)v15, v14, v26, (__int64)v32, 0, v27, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64, __int64))(*(_QWORD *)v13[11] + 16LL))(
    v13[11],
    v17,
    &v29,
    v13[7],
    v13[8],
    v16);
  v18 = *v13;
  v19 = *v13 + 16LL * *((unsigned int *)v13 + 2);
  while ( v19 != v18 )
  {
    v20 = *(_QWORD *)(v18 + 8);
    v21 = *(_DWORD *)v18;
    v18 += 16;
    sub_B99FD0(v17, v21, v20);
  }
  v22 = *(unsigned int ***)(a1 + 32);
  v32[0] = (__int64)"broadcast";
  v33 = 259;
  v23 = *(_DWORD *)(v28 + 32);
  BYTE4(v29) = *(_BYTE *)(v28 + 8) == 18;
  LODWORD(v29) = v23;
  v24 = sub_B37620(v22, (__int64)v29, v17, v32);
  return sub_F162A0(a1, a2, v24);
}
