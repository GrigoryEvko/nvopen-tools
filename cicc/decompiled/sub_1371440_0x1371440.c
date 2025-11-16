// Function: sub_1371440
// Address: 0x1371440
//
__int64 __fastcall sub_1371440(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        const void **a4,
        const void **a5,
        const void **a6,
        const void **a7)
{
  __int64 v10; // r12
  _BYTE *v11; // r9
  _BYTE *v12; // rax
  _BYTE *v13; // r11
  __int64 v14; // r8
  _BYTE *v15; // r10
  void *v16; // rdi
  size_t v17; // r15
  __int64 v18; // r14
  __int64 v19; // rdx
  size_t v20; // r10
  __int64 v21; // rdx
  __int64 v22; // rax
  unsigned __int64 v23; // r14
  int v24; // r14d
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rdx
  int v27; // r14d
  __int64 v28; // rcx
  _QWORD *v29; // rax
  _QWORD *i; // rdx
  _BYTE *v32; // [rsp+0h] [rbp-60h]
  _BYTE *v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  _BYTE *v35; // [rsp+10h] [rbp-50h]
  size_t v36; // [rsp+10h] [rbp-50h]
  _BYTE *v37; // [rsp+18h] [rbp-48h]
  _BYTE *v38; // [rsp+18h] [rbp-48h]
  _BYTE *v39; // [rsp+18h] [rbp-48h]
  _BYTE *v41; // [rsp+20h] [rbp-40h]
  _BYTE *v42; // [rsp+20h] [rbp-40h]
  __int64 v44; // [rsp+28h] [rbp-38h]
  unsigned __int64 v45; // [rsp+28h] [rbp-38h]

  v10 = sub_22077B0(192);
  v11 = *a6;
  v12 = *a5;
  v13 = *a7;
  v14 = v10 + 128;
  v19 = *a3;
  v15 = *a4;
  *(_BYTE *)(v10 + 24) = 0;
  v16 = (void *)(v10 + 128);
  v44 = v10 + 112;
  *(_QWORD *)(v10 + 16) = v19;
  v17 = v12 - v15;
  *(_QWORD *)(v10 + 32) = v10 + 48;
  *(_QWORD *)(v10 + 40) = 0x400000000LL;
  v18 = (v12 - v15) >> 2;
  *(_QWORD *)(v10 + 120) = 0x400000000LL;
  LODWORD(v19) = 0;
  *(_DWORD *)(v10 + 28) = 1;
  *(_QWORD *)(v10 + 112) = v10 + 128;
  if ( (unsigned __int64)(v12 - v15) > 0x10 )
  {
    v32 = v13;
    v33 = v11;
    v35 = v12;
    v38 = v15;
    sub_16CD150(v44, v10 + 128, (v12 - v15) >> 2, 4);
    v19 = *(unsigned int *)(v10 + 120);
    v13 = v32;
    v11 = v33;
    v12 = v35;
    v15 = v38;
    v16 = (void *)(*(_QWORD *)(v10 + 112) + 4 * v19);
    v14 = v10 + 128;
  }
  if ( v15 != v12 )
  {
    v34 = v14;
    v37 = v13;
    v41 = v11;
    memcpy(v16, v15, v17);
    LODWORD(v19) = *(_DWORD *)(v10 + 120);
    v14 = v34;
    v13 = v37;
    v11 = v41;
  }
  *(_QWORD *)(v10 + 168) = 0;
  *(_QWORD *)(v10 + 152) = 0x100000000LL;
  v20 = v13 - v11;
  v21 = (unsigned int)(v19 + v18);
  *(_DWORD *)(v10 + 120) = v21;
  *(_WORD *)(v10 + 184) = 0;
  v22 = *(unsigned int *)(v10 + 124);
  *(_DWORD *)(v10 + 28) = v21;
  v23 = (v13 - v11) >> 2;
  *(_QWORD *)(v10 + 144) = v10 + 160;
  *(_QWORD *)(v10 + 176) = 0;
  if ( v23 > v22 - v21 )
  {
    v36 = v13 - v11;
    v39 = v13;
    v42 = v11;
    sub_16CD150(v44, v14, v23 + v21, 4);
    v21 = *(unsigned int *)(v10 + 120);
    v20 = v36;
    v13 = v39;
    v11 = v42;
  }
  if ( v11 != v13 )
  {
    memcpy((void *)(*(_QWORD *)(v10 + 112) + 4 * v21), v11, v20);
    LODWORD(v21) = *(_DWORD *)(v10 + 120);
  }
  v24 = v21 + v23;
  v25 = *(unsigned int *)(v10 + 152);
  v26 = *(unsigned int *)(v10 + 28);
  *(_DWORD *)(v10 + 120) = v24;
  v27 = v26;
  if ( v26 < v25 )
    goto LABEL_17;
  if ( v26 > v25 )
  {
    if ( v26 > *(unsigned int *)(v10 + 156) )
    {
      v45 = v26;
      sub_16CD150(v10 + 144, v10 + 160, v26, 8);
      v25 = *(unsigned int *)(v10 + 152);
      v26 = v45;
    }
    v28 = *(_QWORD *)(v10 + 144);
    v29 = (_QWORD *)(v28 + 8 * v25);
    for ( i = (_QWORD *)(v28 + 8 * v26); i != v29; ++v29 )
    {
      if ( v29 )
        *v29 = 0;
    }
LABEL_17:
    *(_DWORD *)(v10 + 152) = v27;
  }
  sub_2208C80(v10, a2);
  ++*(_QWORD *)(a1 + 16);
  return v10;
}
