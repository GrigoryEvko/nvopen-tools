// Function: sub_15F5350
// Address: 0x15f5350
//
__int64 __fastcall sub_15F5350(__int64 a1, unsigned int a2, char a3)
{
  __int64 v5; // rdx
  __int64 v6; // rcx
  int v7; // edi
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r10
  __int64 *v11; // rax
  __int64 *v12; // rsi
  __int64 v13; // r13
  __int64 v14; // r11
  unsigned __int64 v15; // rcx
  __int64 i; // rdi
  __int64 v17; // rdx
  __int64 v18; // r8
  unsigned __int64 v19; // rdi
  __int64 v20; // rdi
  __int64 v21; // r9
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rdx
  _QWORD *v25; // rax
  __int64 v26; // rcx
  unsigned __int64 v27; // rdx
  int v28; // eax
  int v29; // edx
  __int64 v31; // r8
  __int64 v32; // rdi
  __int64 v33; // rax

  v5 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v6 = 24 * v5;
  v7 = *(_BYTE *)(a1 + 23) & 0x40;
  if ( v7 )
  {
    v8 = *(_QWORD *)(a1 - 8);
    v9 = a2 + 1LL;
    v10 = 24LL * a2;
    v11 = (__int64 *)(v8 + v10);
    v10 += 24;
    v12 = (__int64 *)(v8 + v10);
    v13 = *v11;
    v14 = v6 - v10;
    v15 = 0xAAAAAAAAAAAAAAABLL * ((v6 - v10) >> 3);
    if ( v14 <= 0 )
      goto LABEL_15;
  }
  else
  {
    v9 = a2 + 1LL;
    v31 = 24LL * a2;
    v11 = (__int64 *)(a1 - v6 + v31);
    v31 += 24;
    v12 = (__int64 *)(a1 - v6 + v31);
    v13 = *v11;
    v32 = v6 - v31;
    v15 = 0xAAAAAAAAAAAAAAABLL * ((v6 - v31) >> 3);
    if ( v32 <= 0 )
    {
LABEL_14:
      LOBYTE(v7) = 0;
      v8 = a1 - 24 * v5;
      goto LABEL_15;
    }
  }
  for ( i = v13; ; i = *v11 )
  {
    v17 = *v12;
    if ( i )
    {
      v18 = v11[1];
      v19 = v11[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v19 = v18;
      if ( v18 )
        *(_QWORD *)(v18 + 16) = *(_QWORD *)(v18 + 16) & 3LL | v19;
    }
    *v11 = v17;
    if ( v17 )
    {
      v20 = *(_QWORD *)(v17 + 8);
      v11[1] = v20;
      if ( v20 )
        *(_QWORD *)(v20 + 16) = (unsigned __int64)(v11 + 1) | *(_QWORD *)(v20 + 16) & 3LL;
      v11[2] = (v17 + 8) | v11[2] & 3;
      *(_QWORD *)(v17 + 8) = v11;
    }
    v12 += 3;
    v11 += 3;
    if ( !--v15 )
      break;
  }
  v5 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  LOBYTE(v7) = *(_BYTE *)(a1 + 23) & 0x40;
  if ( !(_BYTE)v7 )
    goto LABEL_14;
  v8 = *(_QWORD *)(a1 - 8);
LABEL_15:
  v21 = 8 * v9;
  v22 = v8 + 24LL * *(unsigned int *)(a1 + 56) + 8;
  if ( v22 + v21 != v22 + 8 * v5 )
  {
    memmove((void *)(v8 + 24LL * *(unsigned int *)(a1 + 56) + v21), (const void *)(v22 + v21), 8 * v5 - v21);
    LOBYTE(v7) = *(_BYTE *)(a1 + 23) & 0x40;
    v5 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  }
  v23 = 24 * v5;
  v24 = a1 - 24 * v5;
  if ( (_BYTE)v7 )
    v24 = *(_QWORD *)(a1 - 8);
  v25 = (_QWORD *)(v24 + v23 - 24);
  if ( *v25 )
  {
    v26 = v25[1];
    v27 = v25[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v27 = v26;
    if ( v26 )
      *(_QWORD *)(v26 + 16) = *(_QWORD *)(v26 + 16) & 3LL | v27;
  }
  *v25 = 0;
  v28 = *(_DWORD *)(a1 + 20);
  v29 = (v28 + 0xFFFFFFF) & 0xFFFFFFF;
  *(_DWORD *)(a1 + 20) = v29 | v28 & 0xF0000000;
  if ( v29 || !a3 )
    return v13;
  v33 = sub_1599EF0(*(__int64 ***)a1);
  sub_164D160(a1, v33);
  sub_15F20C0((_QWORD *)a1);
  return v13;
}
