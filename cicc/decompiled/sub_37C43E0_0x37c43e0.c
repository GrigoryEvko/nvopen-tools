// Function: sub_37C43E0
// Address: 0x37c43e0
//
void __fastcall sub_37C43E0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  unsigned __int64 i; // rdx
  size_t v10; // rdx
  _BYTE *v11; // rdi
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rcx
  unsigned __int64 v15; // rdi
  char *v16; // r12
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // rdx
  _QWORD *v20; // rdi
  _BYTE *v21; // rdi
  __int64 v22; // rdi
  char *v23; // r12
  _QWORD v24[2]; // [rsp+0h] [rbp-80h] BYREF
  _BYTE *v25; // [rsp+10h] [rbp-70h] BYREF
  __int64 v26; // [rsp+18h] [rbp-68h]
  _BYTE dest[96]; // [rsp+20h] [rbp-60h] BYREF

  v6 = *(_DWORD *)(a1 + 3480);
  if ( !v6 )
    return;
  if ( !a3 || (i = *(_QWORD *)(a3 + 56), a2 != i) )
  {
    for ( i = a2; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
      ;
  }
  v24[0] = i;
  v24[1] = a3;
  v25 = dest;
  v26 = 0x400000000LL;
  if ( v6 > 4 )
  {
    sub_C8D5F0((__int64)&v25, dest, v6, 0x10u, a5, a6);
    v11 = v25;
    v10 = 16LL * *(unsigned int *)(a1 + 3480);
    if ( !v10 )
      goto LABEL_9;
  }
  else
  {
    v10 = 16LL * v6;
    v11 = dest;
  }
  memcpy(v11, *(const void **)(a1 + 3472), v10);
LABEL_9:
  v14 = *(unsigned int *)(a1 + 56);
  v15 = *(unsigned int *)(a1 + 60);
  LODWORD(v26) = v6;
  v16 = (char *)v24;
  v17 = *(_QWORD *)(a1 + 48);
  v18 = v14 + 1;
  v19 = v14;
  if ( v14 + 1 > v15 )
  {
    v22 = a1 + 48;
    if ( v17 > (unsigned __int64)v24 || (v19 = v17 + 96 * v14, (unsigned __int64)v24 >= v19) )
    {
      sub_37C42C0(v22, v18, v19, v14, v12, v13);
      v14 = *(unsigned int *)(a1 + 56);
      v17 = *(_QWORD *)(a1 + 48);
      v19 = v14;
    }
    else
    {
      v23 = (char *)v24 - v17;
      sub_37C42C0(v22, v18, v19, v14, v12, v13);
      v17 = *(_QWORD *)(a1 + 48);
      v14 = *(unsigned int *)(a1 + 56);
      v16 = &v23[v17];
      v19 = v14;
    }
  }
  v20 = (_QWORD *)(v17 + 96 * v14);
  if ( v20 )
  {
    *v20 = *(_QWORD *)v16;
    v20[1] = *((_QWORD *)v16 + 1);
    v20[2] = v20 + 4;
    v20[3] = 0x400000000LL;
    if ( *((_DWORD *)v16 + 6) )
      sub_37B6B90((__int64)(v20 + 2), (__int64)(v16 + 16), v19, v14, v12, v13);
    LODWORD(v19) = *(_DWORD *)(a1 + 56);
  }
  v21 = v25;
  *(_DWORD *)(a1 + 56) = v19 + 1;
  if ( v21 != dest )
    _libc_free((unsigned __int64)v21);
  *(_DWORD *)(a1 + 3480) = 0;
}
