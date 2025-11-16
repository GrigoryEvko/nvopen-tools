// Function: sub_2DD4980
// Address: 0x2dd4980
//
unsigned __int64 __fastcall sub_2DD4980(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r9
  __int64 v6; // r12
  unsigned __int64 *v7; // r15
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rcx
  bool v11; // cf
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r10
  __int64 v14; // rcx
  __int64 v15; // rbx
  __int64 v16; // r10
  unsigned int v17; // r11d
  __int64 v18; // rbx
  unsigned int *v19; // r12
  unsigned int *v20; // r13
  __int64 v21; // rdx
  int v22; // eax
  int v23; // eax
  unsigned __int64 *i; // r12
  unsigned __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // [rsp+8h] [rbp-68h]
  __int64 v29; // [rsp+10h] [rbp-60h]
  __int64 v30; // [rsp+10h] [rbp-60h]
  __int64 v31; // [rsp+18h] [rbp-58h]
  __int64 v32; // [rsp+18h] [rbp-58h]
  unsigned int v33; // [rsp+18h] [rbp-58h]
  unsigned int v34; // [rsp+20h] [rbp-50h]
  unsigned __int64 v35; // [rsp+20h] [rbp-50h]
  __int64 v36; // [rsp+20h] [rbp-50h]
  __int64 v37; // [rsp+28h] [rbp-48h]
  __int64 v38; // [rsp+28h] [rbp-48h]
  unsigned __int64 v39; // [rsp+30h] [rbp-40h]
  __int64 v40; // [rsp+38h] [rbp-38h]

  v5 = a2;
  v6 = a2;
  v7 = (unsigned __int64 *)a1[1];
  v8 = *a1;
  v9 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)((__int64)v7 - *a1) >> 4);
  if ( v9 == 0x199999999999999LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  if ( v9 )
    v10 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(a1[1] - *a1) >> 4);
  v11 = __CFADD__(v10, v9);
  v12 = v10 - 0x3333333333333333LL * ((__int64)(a1[1] - *a1) >> 4);
  v13 = a2 - v8;
  v14 = v11;
  if ( v11 )
  {
    v26 = 0x7FFFFFFFFFFFFFD0LL;
  }
  else
  {
    if ( !v12 )
    {
      v39 = 0;
      v15 = 80;
      v40 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0x199999999999999LL )
      v12 = 0x199999999999999LL;
    v26 = 80 * v12;
  }
  v32 = a3;
  v27 = sub_22077B0(v26);
  v13 = a2 - v8;
  v5 = a2;
  v40 = v27;
  a3 = v32;
  v39 = v27 + v26;
  v15 = v27 + 80;
LABEL_7:
  v16 = v40 + v13;
  if ( v16 )
  {
    a5 = *(_QWORD *)a3;
    *(_QWORD *)v16 = v16 + 16;
    *(_QWORD *)(v16 + 8) = 0x600000000LL;
    v17 = (unsigned int)(a5 + 63) >> 6;
    a3 = v17;
    if ( v17 > 6 )
    {
      v28 = a5;
      v30 = v5;
      v33 = (unsigned int)(a5 + 63) >> 6;
      v36 = v17;
      v38 = v16;
      sub_C8D5F0(v16, (const void *)(v16 + 16), v17, 8u, a5, v5);
      memset(*(void **)v38, 0, 8 * v36);
      v16 = v38;
      v5 = v30;
      a5 = v28;
      *(_DWORD *)(v38 + 8) = v33;
    }
    else
    {
      if ( v17 )
      {
        v29 = a5;
        v31 = v5;
        v34 = (unsigned int)(a5 + 63) >> 6;
        v37 = v16;
        memset((void *)(v16 + 16), 0, 8LL * v17);
        v16 = v37;
        v17 = v34;
        v5 = v31;
        a5 = v29;
      }
      *(_DWORD *)(v16 + 8) = v17;
    }
    *(_DWORD *)(v16 + 64) = a5;
    *(_DWORD *)(v16 + 72) = 1;
  }
  if ( v5 != v8 )
  {
    v18 = v40;
    v19 = (unsigned int *)v8;
    v35 = v8;
    v20 = (unsigned int *)v5;
    while ( 1 )
    {
      if ( v18 )
      {
        *(_DWORD *)(v18 + 8) = 0;
        *(_QWORD *)v18 = v18 + 16;
        *(_DWORD *)(v18 + 12) = 6;
        v21 = v19[2];
        if ( (_DWORD)v21 )
          sub_2DD32C0(v18, (__int64)v19, v21, v14, a5, v5);
        *(_DWORD *)(v18 + 64) = v19[16];
        *(_DWORD *)(v18 + 72) = v19[18];
      }
      v19 += 20;
      a3 = v18 + 80;
      if ( v19 == v20 )
        break;
      v18 += 80;
    }
    v5 = (__int64)v20;
    v6 = a2;
    v8 = v35;
    v15 = v18 + 160;
  }
  if ( (unsigned __int64 *)v5 != v7 )
  {
    do
    {
      *(_DWORD *)(v15 + 8) = 0;
      *(_QWORD *)v15 = v15 + 16;
      v23 = *(_DWORD *)(v6 + 8);
      *(_DWORD *)(v15 + 12) = 6;
      if ( v23 )
        sub_2DD32C0(v15, v6, a3, v14, a5, v5);
      v22 = *(_DWORD *)(v6 + 64);
      v6 += 80;
      v15 += 80;
      *(_DWORD *)(v15 - 16) = v22;
      *(_DWORD *)(v15 - 8) = *(_DWORD *)(v6 - 8);
    }
    while ( (unsigned __int64 *)v6 != v7 );
  }
  for ( i = (unsigned __int64 *)v8; v7 != i; i += 10 )
  {
    if ( (unsigned __int64 *)*i != i + 2 )
      _libc_free(*i);
  }
  if ( v8 )
    j_j___libc_free_0(v8);
  a1[1] = v15;
  *a1 = v40;
  a1[2] = v39;
  return v39;
}
