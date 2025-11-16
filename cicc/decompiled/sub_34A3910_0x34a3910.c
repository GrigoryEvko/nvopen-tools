// Function: sub_34A3910
// Address: 0x34a3910
//
_QWORD *__fastcall sub_34A3910(__int64 a1, __int64 a2, __int64 a3)
{
  char v5; // cl
  int v6; // ecx
  __int64 v7; // rdi
  __int64 v8; // rsi
  unsigned int v9; // edx
  _QWORD *v10; // rax
  __int64 v11; // r9
  _QWORD *v12; // r8
  unsigned __int64 *v13; // rbx
  unsigned int v15; // eax
  int v16; // edx
  unsigned int v17; // edi
  __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r12
  _QWORD *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  unsigned __int64 v24; // r9
  unsigned __int64 v25; // r12
  int v26; // r11d
  __int64 *v27; // r10
  __int64 v28[2]; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v29; // [rsp+18h] [rbp-28h] BYREF

  v5 = *(_BYTE *)(a3 + 8);
  v28[0] = a2;
  v6 = v5 & 1;
  if ( v6 )
  {
    v7 = a3 + 16;
    v8 = 3;
  }
  else
  {
    v8 = *(unsigned int *)(a3 + 24);
    v7 = *(_QWORD *)(a3 + 16);
    if ( !(_DWORD)v8 )
    {
      v15 = *(_DWORD *)(a3 + 8);
      ++*(_QWORD *)a3;
      v29 = 0;
      v16 = (v15 >> 1) + 1;
LABEL_9:
      v17 = 3 * v8;
      goto LABEL_10;
    }
    v8 = (unsigned int)(v8 - 1);
  }
  v9 = v8 & ((LODWORD(v28[0]) >> 9) ^ (LODWORD(v28[0]) >> 4));
  v10 = (_QWORD *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( v28[0] == *v10 )
  {
LABEL_4:
    v12 = (_QWORD *)v10[1];
    v13 = v10 + 1;
    if ( v12 )
      return v12;
    goto LABEL_15;
  }
  v26 = 1;
  v27 = 0;
  while ( v11 != -4096 )
  {
    if ( !v27 && v11 == -8192 )
      v27 = v10;
    v9 = v8 & (v26 + v9);
    v10 = (_QWORD *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( v28[0] == *v10 )
      goto LABEL_4;
    ++v26;
  }
  v17 = 12;
  v8 = 4;
  if ( !v27 )
    v27 = v10;
  v15 = *(_DWORD *)(a3 + 8);
  ++*(_QWORD *)a3;
  v29 = v27;
  v16 = (v15 >> 1) + 1;
  if ( !(_BYTE)v6 )
  {
    v8 = *(unsigned int *)(a3 + 24);
    goto LABEL_9;
  }
LABEL_10:
  if ( v17 <= 4 * v16 )
  {
    LODWORD(v8) = 2 * v8;
    goto LABEL_20;
  }
  if ( (int)v8 - *(_DWORD *)(a3 + 12) - v16 <= (unsigned int)v8 >> 3 )
  {
LABEL_20:
    sub_34A34A0(a3, v8);
    v8 = (__int64)v28;
    sub_34A1A80(a3, v28, &v29);
    v15 = *(_DWORD *)(a3 + 8);
  }
  *(_DWORD *)(a3 + 8) = (2 * (v15 >> 1) + 2) | v15 & 1;
  v18 = v29;
  if ( *v29 != -4096 )
    --*(_DWORD *)(a3 + 12);
  v19 = v28[0];
  v18[1] = 0;
  v13 = (unsigned __int64 *)(v18 + 1);
  *v18 = v19;
LABEL_15:
  v20 = a1 + 376;
  v21 = (_QWORD *)sub_22077B0(0xD8u);
  v12 = v21;
  if ( v21 )
  {
    *v21 = v20;
    v21[26] = v20;
    memset(v21 + 1, 0, 0xC0u);
    v23 = 0;
    v21[25] = 0;
  }
  v25 = *v13;
  *v13 = (unsigned __int64)v21;
  if ( v25 )
  {
    sub_34A2530((unsigned int *)(v25 + 8), v8, v22, v23, (unsigned __int64)v21, v24);
    j_j___libc_free_0(v25);
    return (_QWORD *)*v13;
  }
  return v12;
}
