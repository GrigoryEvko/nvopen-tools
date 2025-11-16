// Function: sub_154E670
// Address: 0x154e670
//
void __fastcall sub_154E670(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  unsigned int v5; // esi
  __int64 v6; // r8
  int v7; // r10d
  __int64 *v8; // rcx
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rdi
  int v13; // edx
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // r14
  _BYTE *v17; // rsi
  int v18; // eax
  __int64 *v19; // [rsp-40h] [rbp-40h] BYREF
  __int64 v20; // [rsp-38h] [rbp-38h] BYREF
  int v21; // [rsp-30h] [rbp-30h]

  if ( *(_BYTE *)a2 == 6 )
    return;
  v3 = a1 + 112;
  v20 = a2;
  v5 = *(_DWORD *)(a1 + 136);
  v21 = *(_DWORD *)(a1 + 144);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 112);
LABEL_7:
    v5 *= 2;
LABEL_8:
    sub_154E4B0(v3, v5);
    sub_154CD30(v3, &v20, &v19);
    v8 = v19;
    v12 = v20;
    v13 = *(_DWORD *)(a1 + 128) + 1;
    goto LABEL_9;
  }
  v6 = *(_QWORD *)(a1 + 120);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
    return;
  while ( v11 != -8 )
  {
    if ( v8 || v11 != -16 )
      v10 = v8;
    v9 = (v5 - 1) & (v7 + v9);
    v11 = *(_QWORD *)(v6 + 16LL * v9);
    if ( a2 == v11 )
      return;
    ++v7;
    v8 = v10;
    v10 = (__int64 *)(v6 + 16LL * v9);
  }
  if ( !v8 )
    v8 = v10;
  v18 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  v13 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v5 )
    goto LABEL_7;
  v12 = a2;
  if ( v5 - *(_DWORD *)(a1 + 132) - v13 <= v5 >> 3 )
    goto LABEL_8;
LABEL_9:
  *(_DWORD *)(a1 + 128) = v13;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 132);
  *v8 = v12;
  *((_DWORD *)v8 + 2) = v21;
  ++*(_DWORD *)(a1 + 144);
  v14 = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)v14 )
  {
    v15 = 0;
    v16 = *(unsigned int *)(a2 + 8);
    v17 = *(_BYTE **)(a2 - 8 * v14);
    if ( !v17 )
      goto LABEL_15;
LABEL_13:
    if ( (unsigned __int8)(*v17 - 4) <= 0x1Eu )
      sub_154E670(a1);
LABEL_15:
    while ( v16 != ++v15 )
    {
      v17 = *(_BYTE **)(a2 + 8 * (v15 - *(unsigned int *)(a2 + 8)));
      if ( v17 )
        goto LABEL_13;
    }
  }
}
