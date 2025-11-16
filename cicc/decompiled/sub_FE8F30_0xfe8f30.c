// Function: sub_FE8F30
// Address: 0xfe8f30
//
__int64 __fastcall sub_FE8F30(
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
  const void *v14; // r8
  _BYTE *v15; // r10
  void *v16; // rdi
  size_t v17; // r15
  __int64 v18; // r14
  __int64 v19; // rdx
  unsigned int v20; // r14d
  unsigned __int64 v21; // rcx
  size_t v22; // r10
  __int64 v23; // rax
  __int64 v24; // r14
  int v25; // r14d
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // r14
  unsigned int v28; // r8d
  __int64 v29; // rdx
  _QWORD *v30; // rax
  _QWORD *i; // rdx
  _BYTE *v33; // [rsp+0h] [rbp-60h]
  _BYTE *v34; // [rsp+8h] [rbp-58h]
  const void *v35; // [rsp+10h] [rbp-50h]
  _BYTE *v36; // [rsp+10h] [rbp-50h]
  size_t v37; // [rsp+10h] [rbp-50h]
  _BYTE *v38; // [rsp+18h] [rbp-48h]
  _BYTE *v39; // [rsp+18h] [rbp-48h]
  _BYTE *v40; // [rsp+18h] [rbp-48h]
  _BYTE *v42; // [rsp+20h] [rbp-40h]
  _BYTE *v43; // [rsp+20h] [rbp-40h]
  __int64 v45; // [rsp+28h] [rbp-38h]
  unsigned int v46; // [rsp+28h] [rbp-38h]

  v10 = sub_22077B0(192);
  v11 = *a6;
  v12 = *a5;
  v13 = *a7;
  v14 = (const void *)(v10 + 128);
  v19 = *a3;
  v15 = *a4;
  *(_BYTE *)(v10 + 24) = 0;
  v16 = (void *)(v10 + 128);
  v45 = v10 + 112;
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
    v33 = v13;
    v34 = v11;
    v36 = v12;
    v39 = v15;
    sub_C8D5F0(v45, (const void *)(v10 + 128), (v12 - v15) >> 2, 4u, (__int64)v14, (__int64)v11);
    v19 = *(unsigned int *)(v10 + 120);
    v13 = v33;
    v11 = v34;
    v12 = v36;
    v15 = v39;
    v16 = (void *)(*(_QWORD *)(v10 + 112) + 4 * v19);
    v14 = (const void *)(v10 + 128);
  }
  if ( v15 != v12 )
  {
    v35 = v14;
    v38 = v13;
    v42 = v11;
    memcpy(v16, v15, v17);
    LODWORD(v19) = *(_DWORD *)(v10 + 120);
    v14 = v35;
    v13 = v38;
    v11 = v42;
  }
  v20 = v19 + v18;
  v21 = *(unsigned int *)(v10 + 124);
  v22 = v13 - v11;
  *(_QWORD *)(v10 + 152) = 0x100000000LL;
  *(_DWORD *)(v10 + 120) = v20;
  *(_WORD *)(v10 + 184) = 0;
  v23 = v20;
  *(_DWORD *)(v10 + 28) = v20;
  v24 = (v13 - v11) >> 2;
  *(_QWORD *)(v10 + 144) = v10 + 160;
  *(_QWORD *)(v10 + 168) = 0;
  *(_QWORD *)(v10 + 176) = 0;
  if ( v24 + v23 > v21 )
  {
    v37 = v13 - v11;
    v40 = v13;
    v43 = v11;
    sub_C8D5F0(v45, v14, v24 + v23, 4u, (__int64)v14, (__int64)v11);
    v23 = *(unsigned int *)(v10 + 120);
    v22 = v37;
    v13 = v40;
    v11 = v43;
  }
  if ( v11 != v13 )
  {
    memcpy((void *)(*(_QWORD *)(v10 + 112) + 4 * v23), v11, v22);
    LODWORD(v23) = *(_DWORD *)(v10 + 120);
  }
  v25 = v23 + v24;
  v26 = *(unsigned int *)(v10 + 152);
  *(_DWORD *)(v10 + 120) = v25;
  v27 = *(unsigned int *)(v10 + 28);
  v28 = *(_DWORD *)(v10 + 28);
  if ( v27 != v26 )
  {
    if ( v27 >= v26 )
    {
      if ( v27 > *(unsigned int *)(v10 + 156) )
      {
        v46 = *(_DWORD *)(v10 + 28);
        sub_C8D5F0(v10 + 144, (const void *)(v10 + 160), v46, 8u, v27, (__int64)v11);
        v26 = *(unsigned int *)(v10 + 152);
        v28 = v46;
      }
      v29 = *(_QWORD *)(v10 + 144);
      v30 = (_QWORD *)(v29 + 8 * v26);
      for ( i = (_QWORD *)(v29 + 8 * v27); i != v30; ++v30 )
      {
        if ( v30 )
          *v30 = 0;
      }
    }
    *(_DWORD *)(v10 + 152) = v28;
  }
  sub_2208C80(v10, a2);
  ++*(_QWORD *)(a1 + 16);
  return v10;
}
