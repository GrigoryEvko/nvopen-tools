// Function: sub_B00B60
// Address: 0xb00b60
//
__int64 __fastcall sub_B00B60(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rcx
  __int64 v6; // r13
  __int64 v7; // r9
  int v8; // r13d
  unsigned int v9; // eax
  size_t v10; // r9
  size_t v11; // rdx
  int v12; // r10d
  __int64 v13; // rcx
  unsigned int v14; // r8d
  size_t v15; // r15
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  int v20; // eax
  _QWORD *v21; // rax
  __int64 v22; // r15
  _QWORD *v23; // rdi
  _QWORD *v24; // rax
  size_t v25; // r13
  void *v26; // r8
  __int64 v27; // rbx
  __int64 v28; // rbx
  unsigned int v29; // esi
  int v30; // eax
  _QWORD *v31; // rdx
  int v32; // eax
  size_t v33; // [rsp+8h] [rbp-78h]
  int v34; // [rsp+14h] [rbp-6Ch]
  size_t v35; // [rsp+18h] [rbp-68h]
  unsigned int v36; // [rsp+18h] [rbp-68h]
  size_t na; // [rsp+20h] [rbp-60h]
  size_t n; // [rsp+20h] [rbp-60h]
  __int64 v39; // [rsp+28h] [rbp-58h]
  __int64 v40; // [rsp+28h] [rbp-58h]
  __int64 v41; // [rsp+38h] [rbp-48h] BYREF
  _QWORD *v42; // [rsp+40h] [rbp-40h] BYREF
  _QWORD *v43; // [rsp+48h] [rbp-38h] BYREF

  v5 = *a1;
  v6 = *(unsigned int *)(*a1 + 656);
  v39 = *a1;
  v7 = *(_QWORD *)(*a1 + 640);
  if ( !(_DWORD)v6 )
  {
    v17 = 0;
    v18 = *(_QWORD *)(v5 + 640);
    goto LABEL_8;
  }
  v35 = *(_QWORD *)(v5 + 640);
  v8 = v6 - 1;
  na = 8 * a3;
  v9 = sub_AF6940(a2, (__int64)&a2[a3]);
  v10 = v35;
  v11 = na;
  v12 = 1;
  v13 = v39;
  v14 = v8 & v9;
  v15 = v35 + 8LL * (v8 & v9);
  v16 = *(_QWORD *)v15;
  if ( *(_QWORD *)v15 == -4096 )
  {
LABEL_23:
    v7 = *(_QWORD *)(v13 + 640);
    v6 = *(unsigned int *)(v13 + 656);
    v18 = *(_QWORD *)(*a1 + 640);
    v17 = *(unsigned int *)(*a1 + 656);
LABEL_8:
    v15 = v7 + 8 * v6;
    if ( v15 == v18 + 8 * v17 )
      goto LABEL_13;
    return *(_QWORD *)v15;
  }
  while ( 1 )
  {
    if ( v16 != -8192 && a3 == *(_DWORD *)(v16 + 144) )
    {
      v34 = v12;
      v36 = v14;
      n = v10;
      v40 = v13;
      if ( !v11 )
        break;
      v33 = v11;
      v20 = memcmp(a2, *(const void **)(v16 + 136), v11);
      v11 = v33;
      v13 = v40;
      v10 = n;
      v14 = v36;
      v12 = v34;
      if ( !v20 )
        break;
    }
    v14 = v8 & (v12 + v14);
    v15 = v10 + 8LL * v14;
    v16 = *(_QWORD *)v15;
    if ( *(_QWORD *)v15 == -4096 )
      goto LABEL_23;
    ++v12;
  }
  if ( v15 != *(_QWORD *)(*a1 + 640) + 8LL * *(unsigned int *)(*a1 + 656) )
    return *(_QWORD *)v15;
LABEL_13:
  v21 = (_QWORD *)sub_22077B0(184);
  v22 = (__int64)v21;
  if ( !v21 )
    goto LABEL_21;
  *v21 = 4;
  v23 = v21 + 5;
  v24 = v21 + 17;
  *(v24 - 16) = a1;
  *(v24 - 15) = 0;
  *(v24 - 14) = 0;
  *(v24 - 13) = 1;
  do
  {
    if ( v23 )
      *v23 = -4096;
    v23 += 3;
  }
  while ( v23 != v24 );
  v25 = 8 * a3;
  v26 = (void *)(v22 + 152);
  *(_QWORD *)(v22 + 136) = v22 + 152;
  *(_QWORD *)(v22 + 144) = 0x400000000LL;
  v27 = (8 * a3) >> 3;
  if ( v25 > 0x20 )
  {
    sub_C8D5F0(v23, v22 + 152, v27, 8);
    v26 = (void *)(*(_QWORD *)(v22 + 136) + 8LL * *(unsigned int *)(v22 + 144));
    goto LABEL_25;
  }
  if ( v25 )
  {
LABEL_25:
    memcpy(v26, a2, v25);
    LODWORD(v25) = *(_DWORD *)(v22 + 144);
  }
  *(_DWORD *)(v22 + 144) = v25 + v27;
  sub_AF5060(v22);
LABEL_21:
  v28 = *a1;
  v41 = v22;
  if ( !(unsigned __int8)sub_AFC070(v28 + 632, &v41, &v42) )
  {
    v29 = *(_DWORD *)(v28 + 656);
    v30 = *(_DWORD *)(v28 + 648);
    v31 = v42;
    ++*(_QWORD *)(v28 + 632);
    v32 = v30 + 1;
    v43 = v31;
    if ( 4 * v32 >= 3 * v29 )
    {
      v29 *= 2;
    }
    else if ( v29 - *(_DWORD *)(v28 + 652) - v32 > v29 >> 3 )
    {
LABEL_28:
      *(_DWORD *)(v28 + 648) = v32;
      if ( *v31 != -4096 )
        --*(_DWORD *)(v28 + 652);
      *v31 = v41;
      return v41;
    }
    sub_B00970(v28 + 632, v29);
    sub_AFC070(v28 + 632, &v41, &v43);
    v31 = v43;
    v32 = *(_DWORD *)(v28 + 648) + 1;
    goto LABEL_28;
  }
  return v41;
}
