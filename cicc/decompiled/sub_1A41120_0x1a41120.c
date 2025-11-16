// Function: sub_1A41120
// Address: 0x1a41120
//
unsigned __int64 *__fastcall sub_1A41120(
        __int64 a1,
        unsigned __int64 a2,
        _QWORD *a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v14; // r15
  __int64 v15; // r13
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  __int64 *v19; // rax
  __int64 v20; // rsi
  unsigned __int64 v21; // rcx
  __int64 v22; // rcx
  int v23; // r9d
  unsigned __int64 *v24; // rax
  __int64 v25; // r13
  unsigned __int64 *v26; // r8
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // r13
  __int64 v30; // r15
  _QWORD *v31; // r13
  double v32; // xmm4_8
  double v33; // xmm5_8
  int v34; // r8d
  int v35; // r9d
  __int64 v36; // rax
  unsigned __int64 *result; // rax
  unsigned __int64 *v38; // r15
  unsigned __int64 *v39; // rax
  unsigned __int64 *v40; // rax
  __int64 v41; // rdx
  unsigned __int64 *v42; // r15
  _BOOL8 v43; // rdi
  __int64 v44; // [rsp+8h] [rbp-48h]
  __int64 v45; // [rsp+10h] [rbp-40h]
  unsigned __int64 *v46; // [rsp+18h] [rbp-38h]
  unsigned __int64 *v47; // [rsp+18h] [rbp-38h]

  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
  {
    v14 = 0;
    v15 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    do
    {
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v16 = *(_QWORD *)(a2 - 8);
      else
        v16 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v17 = sub_1599EF0(**(__int64 ****)(v16 + v14));
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
        v18 = *(_QWORD *)(a2 - 8);
      else
        v18 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v19 = (__int64 *)(v14 + v18);
      if ( *v19 )
      {
        v20 = v19[1];
        v21 = v19[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v21 = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = *(_QWORD *)(v20 + 16) & 3LL | v21;
      }
      *v19 = v17;
      if ( v17 )
      {
        v22 = *(_QWORD *)(v17 + 8);
        v19[1] = v22;
        if ( v22 )
          *(_QWORD *)(v22 + 16) = (unsigned __int64)(v19 + 1) | *(_QWORD *)(v22 + 16) & 3LL;
        v19[2] = (v17 + 8) | v19[2] & 3;
        *(_QWORD *)(v17 + 8) = v19;
      }
      v14 += 24;
    }
    while ( v15 != v14 );
  }
  sub_1A3F630(a1, a2, (__int64)a3);
  v24 = *(unsigned __int64 **)(a1 + 176);
  v25 = a1 + 168;
  v26 = (unsigned __int64 *)(a1 + 168);
  if ( !v24 )
    goto LABEL_32;
  do
  {
    while ( 1 )
    {
      v27 = v24[2];
      v28 = v24[3];
      if ( v24[4] >= a2 )
        break;
      v24 = (unsigned __int64 *)v24[3];
      if ( !v28 )
        goto LABEL_22;
    }
    v26 = v24;
    v24 = (unsigned __int64 *)v24[2];
  }
  while ( v27 );
LABEL_22:
  if ( v26 == (unsigned __int64 *)v25 || v26[4] > a2 )
  {
LABEL_32:
    v38 = v26;
    v39 = (unsigned __int64 *)sub_22077B0(120);
    v39[4] = a2;
    v39[5] = (unsigned __int64)(v39 + 7);
    v39[6] = 0x800000000LL;
    v47 = v39;
    v40 = sub_1A41020((_QWORD *)(a1 + 160), v38, v39 + 4);
    v42 = v40;
    if ( v41 )
    {
      v43 = v40 || v25 == v41 || a2 < *(_QWORD *)(v41 + 32);
      sub_220F040(v43, v47, v41, a1 + 168);
      v26 = v47;
      ++*(_QWORD *)(a1 + 200);
    }
    else
    {
      j_j___libc_free_0(v47, 120);
      v26 = v42;
    }
  }
  v29 = *((unsigned int *)v26 + 12);
  v44 = (__int64)(v26 + 5);
  if ( (_DWORD)v29 )
  {
    v30 = 0;
    v45 = 8 * v29;
    do
    {
      v31 = *(_QWORD **)(v26[5] + v30);
      if ( v31 )
      {
        v46 = v26;
        sub_164B7C0(*(_QWORD *)(*a3 + v30), (__int64)v31);
        sub_164D160((__int64)v31, *(_QWORD *)(*a3 + v30), a4, a5, a6, a7, v32, v33, a10, a11);
        sub_15F20C0(v31);
        v26 = v46;
      }
      v30 += 8;
    }
    while ( v30 != v45 );
  }
  sub_1A3EC80(v44, (__int64)a3, v28, v27, (int)v26, v23);
  v36 = *(unsigned int *)(a1 + 216);
  if ( (unsigned int)v36 >= *(_DWORD *)(a1 + 220) )
  {
    sub_16CD150(a1 + 208, (const void *)(a1 + 224), 0, 16, v34, v35);
    v36 = *(unsigned int *)(a1 + 216);
  }
  result = (unsigned __int64 *)(*(_QWORD *)(a1 + 208) + 16 * v36);
  *result = a2;
  result[1] = v44;
  ++*(_DWORD *)(a1 + 216);
  return result;
}
