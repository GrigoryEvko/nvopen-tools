// Function: sub_2670A70
// Address: 0x2670a70
//
__int64 __fastcall sub_2670A70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  unsigned __int64 v7; // r15
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rax
  int v11; // r14d
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // r15
  void *v15; // rdi
  size_t v16; // rdx
  __int64 v17; // r14
  size_t v19; // rdx
  void *v20; // rdi
  unsigned __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rbx
  __int64 v24; // rax
  unsigned int v25; // eax
  unsigned __int8 *v26; // r11
  unsigned __int8 *v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rcx
  __int64 v30; // rax
  unsigned int v31; // [rsp+4h] [rbp-7Ch]
  __int64 v32; // [rsp+8h] [rbp-78h]
  unsigned __int8 *v33; // [rsp+10h] [rbp-70h]
  __int64 v34; // [rsp+20h] [rbp-60h]
  __int64 v35; // [rsp+20h] [rbp-60h]
  __int64 v36; // [rsp+20h] [rbp-60h]
  __int64 v37; // [rsp+30h] [rbp-50h]
  unsigned __int64 v39; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v40; // [rsp+48h] [rbp-38h]

  v6 = a1;
  v7 = *(_QWORD *)(*(_QWORD *)(a2 + 72) + 32LL);
  if ( v7 > *(unsigned int *)(a1 + 20) )
  {
    *(_DWORD *)(a1 + 16) = 0;
    sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), v7, 8u, a5, a6);
    v20 = *(void **)(a1 + 8);
    v19 = 8 * v7;
    if ( 8 * v7 )
      goto LABEL_20;
LABEL_7:
    v10 = *(unsigned int *)(v6 + 100);
    *(_DWORD *)(v6 + 16) = v7;
    v11 = v7;
    if ( v7 > v10 )
      goto LABEL_21;
    goto LABEL_8;
  }
  v8 = *(unsigned int *)(a1 + 16);
  v9 = v8;
  if ( v7 <= v8 )
    v9 = *(_QWORD *)(*(_QWORD *)(a2 + 72) + 32LL);
  if ( v9 )
  {
    memset(*(void **)(a1 + 8), 0, 8 * v9);
    v8 = *(unsigned int *)(a1 + 16);
  }
  if ( v7 <= v8 )
    goto LABEL_7;
  if ( v7 == v8 )
    goto LABEL_7;
  v19 = 8 * (v7 - v8);
  v20 = (void *)(*(_QWORD *)(a1 + 8) + 8 * v8);
  if ( !v19 )
    goto LABEL_7;
LABEL_20:
  v11 = v7;
  memset(v20, 0, v19);
  v21 = *(unsigned int *)(v6 + 100);
  *(_DWORD *)(v6 + 16) = v7;
  if ( v7 > v21 )
  {
LABEL_21:
    *(_DWORD *)(v6 + 96) = 0;
    sub_C8D5F0(v6 + 88, (const void *)(v6 + 104), v7, 8u, a5, a6);
    v15 = *(void **)(v6 + 88);
    v16 = 8 * v7;
    if ( !(8 * v7) )
      goto LABEL_16;
    goto LABEL_15;
  }
LABEL_8:
  v12 = *(unsigned int *)(v6 + 96);
  v13 = v12;
  if ( v7 <= v12 )
    v13 = v7;
  if ( v13 )
  {
    memset(*(void **)(v6 + 88), 0, 8 * v13);
    v12 = *(unsigned int *)(v6 + 96);
  }
  if ( v12 >= v7 )
    goto LABEL_16;
  v14 = v7 - v12;
  if ( !v14 )
    goto LABEL_16;
  v15 = (void *)(*(_QWORD *)(v6 + 88) + 8 * v12);
  v16 = 8 * v14;
  if ( !(8 * v14) )
    goto LABEL_16;
LABEL_15:
  memset(v15, 0, v16);
LABEL_16:
  *(_DWORD *)(v6 + 96) = v11;
  v17 = *(_QWORD *)(a2 + 40);
  if ( v17 != *(_QWORD *)(a3 + 40) )
    return 0;
  v22 = sub_B43CC0(a2);
  v31 = sub_AE4380(v22, 0);
  v37 = v17 + 48;
  if ( *(_QWORD *)(v17 + 56) == v17 + 48 )
    goto LABEL_38;
  v32 = v6;
  v23 = *(_QWORD *)(v17 + 56);
  while ( 1 )
  {
    if ( !v23 )
      BUG();
    if ( v23 - 24 == a3 )
      break;
    if ( *(_BYTE *)(v23 - 24) != 62 )
      goto LABEL_28;
    v34 = *(_QWORD *)(v23 - 56);
    v25 = sub_AE43F0(v22, *(_QWORD *)(v34 + 8));
    v26 = (unsigned __int8 *)v34;
    v40 = v25;
    if ( v25 > 0x40 )
    {
      sub_C43690((__int64)&v39, 0, 0);
      v26 = (unsigned __int8 *)v34;
    }
    else
    {
      v39 = 0;
    }
    v27 = sub_BD45C0(v26, v22, (__int64)&v39, 1, 0, 0, 0, 0);
    if ( v40 <= 0x40 )
    {
      v24 = 0;
      if ( v40 )
        v24 = (__int64)(v39 << (64 - (unsigned __int8)v40)) >> (64 - (unsigned __int8)v40);
      if ( (unsigned __int8 *)a2 == v27 )
        goto LABEL_36;
LABEL_28:
      v23 = *(_QWORD *)(v23 + 8);
      if ( v37 == v23 )
        break;
    }
    else
    {
      v33 = v27;
      v35 = *(_QWORD *)v39;
      j_j___libc_free_0_0(v39);
      v24 = v35;
      if ( (unsigned __int8 *)a2 != v33 )
        goto LABEL_28;
LABEL_36:
      v36 = v24 / v31;
      *(_QWORD *)(*(_QWORD *)(v32 + 8) + 8 * v36) = sub_98ACB0(*(unsigned __int8 **)(v23 - 88), 6u);
      *(_QWORD *)(*(_QWORD *)(v32 + 88) + 8 * v36) = v23 - 24;
      v23 = *(_QWORD *)(v23 + 8);
      if ( v37 == v23 )
        break;
    }
  }
  v6 = v32;
LABEL_38:
  v28 = *(unsigned int *)(v6 + 16);
  if ( !(_DWORD)v28 )
    return 1;
  v29 = 8 * v28;
  v30 = 0;
  while ( *(_QWORD *)(*(_QWORD *)(v6 + 8) + v30) && *(_QWORD *)(*(_QWORD *)(v6 + 88) + v30) )
  {
    v30 += 8;
    if ( v29 == v30 )
      return 1;
  }
  return 0;
}
