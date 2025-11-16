// Function: sub_2731260
// Address: 0x2731260
//
unsigned __int64 __fastcall sub_2731260(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 *v5; // r14
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdi
  __int64 v9; // r8
  __int64 v10; // r15
  bool v11; // cf
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r9
  __int64 v14; // rbx
  __int64 v15; // r9
  int v16; // esi
  __int64 v17; // rdi
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rax
  int v21; // eax
  unsigned __int64 *i; // r15
  unsigned __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-58h]
  __int64 v27; // [rsp+8h] [rbp-58h]
  unsigned __int64 v28; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+18h] [rbp-48h]
  __int64 v30; // [rsp+18h] [rbp-48h]
  __int64 v31; // [rsp+20h] [rbp-40h]
  __int64 v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+28h] [rbp-38h]

  v5 = (unsigned __int64 *)a1[1];
  v6 = *a1;
  v7 = 0xCF3CF3CF3CF3CF3DLL * ((__int64)((__int64)v5 - *a1) >> 3);
  if ( v7 == 0xC30C30C30C30C3LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = 1;
  v9 = a2;
  if ( v7 )
    v8 = 0xCF3CF3CF3CF3CF3DLL * ((__int64)((__int64)v5 - v6) >> 3);
  v10 = a2;
  v11 = __CFADD__(v8, v7);
  v12 = v8 - 0x30C30C30C30C30C3LL * ((__int64)((__int64)v5 - v6) >> 3);
  v13 = a2 - v6;
  if ( v11 )
  {
    v24 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v12 )
    {
      v28 = 0;
      v14 = 168;
      v33 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0xC30C30C30C30C3LL )
      v12 = 0xC30C30C30C30C3LL;
    v24 = 168 * v12;
  }
  v26 = a3;
  v25 = sub_22077B0(v24);
  v13 = a2 - v6;
  v9 = a2;
  v33 = v25;
  a3 = v26;
  v28 = v25 + v24;
  v14 = v25 + 168;
LABEL_7:
  v15 = v33 + v13;
  if ( v15 )
  {
    a4 = *(unsigned int *)(a3 + 8);
    *(_QWORD *)v15 = v15 + 16;
    *(_QWORD *)(v15 + 8) = 0x800000000LL;
    if ( (_DWORD)a4 )
    {
      v27 = v9;
      v30 = a3;
      v32 = v15;
      sub_272D8A0(v15, (char **)a3, a3, a4, v9, v15);
      v9 = v27;
      a3 = v30;
      v15 = v32;
    }
    v16 = *(_DWORD *)(a3 + 160);
    *(_QWORD *)(v15 + 144) = *(_QWORD *)(a3 + 144);
    v17 = *(_QWORD *)(a3 + 152);
    *(_DWORD *)(v15 + 160) = v16;
    *(_QWORD *)(v15 + 152) = v17;
  }
  if ( v9 != v6 )
  {
    v18 = v33;
    v19 = v6;
    while ( 1 )
    {
      if ( v18 )
      {
        *(_DWORD *)(v18 + 8) = 0;
        *(_QWORD *)v18 = v18 + 16;
        *(_DWORD *)(v18 + 12) = 8;
        a3 = *(unsigned int *)(v19 + 8);
        if ( (_DWORD)a3 )
        {
          v29 = v9;
          v31 = v19;
          sub_272D7C0(v18, v19, a3, a4, v9, v15);
          v9 = v29;
          v19 = v31;
        }
        *(_QWORD *)(v18 + 144) = *(_QWORD *)(v19 + 144);
        *(_QWORD *)(v18 + 152) = *(_QWORD *)(v19 + 152);
        *(_DWORD *)(v18 + 160) = *(_DWORD *)(v19 + 160);
      }
      v19 += 168;
      if ( v9 == v19 )
        break;
      v18 += 168;
    }
    v14 = v18 + 336;
  }
  if ( (unsigned __int64 *)v9 != v5 )
  {
    do
    {
      *(_DWORD *)(v14 + 8) = 0;
      *(_QWORD *)v14 = v14 + 16;
      v21 = *(_DWORD *)(v10 + 8);
      *(_DWORD *)(v14 + 12) = 8;
      if ( v21 )
        sub_272D7C0(v14, v10, a3, a4, v9, v15);
      v20 = *(_QWORD *)(v10 + 144);
      v10 += 168;
      v14 += 168;
      *(_QWORD *)(v14 - 24) = v20;
      *(_QWORD *)(v14 - 16) = *(_QWORD *)(v10 - 16);
      *(_DWORD *)(v14 - 8) = *(_DWORD *)(v10 - 8);
    }
    while ( v5 != (unsigned __int64 *)v10 );
  }
  for ( i = (unsigned __int64 *)v6; v5 != i; i += 21 )
  {
    if ( (unsigned __int64 *)*i != i + 2 )
      _libc_free(*i);
  }
  if ( v6 )
    j_j___libc_free_0(v6);
  a1[1] = v14;
  *a1 = v33;
  a1[2] = v28;
  return v28;
}
