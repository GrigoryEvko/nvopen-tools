// Function: sub_13763B0
// Address: 0x13763b0
//
__int64 __fastcall sub_13763B0(__int64 *a1, __int64 a2, _DWORD *a3)
{
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rcx
  bool v9; // cf
  unsigned __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r14
  __int64 v13; // rcx
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // r15
  __int64 *v18; // rdi
  __int64 v20; // rsi
  __int64 v21; // rax
  _DWORD *v22; // [rsp+8h] [rbp-48h]
  __int64 v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+18h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = 0x2E8BA2E8BA2E8BA3LL * ((v4 - *a1) >> 3);
  if ( v6 == 0x1745D1745D1745DLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v7 = 1;
  if ( v6 )
    v7 = 0x2E8BA2E8BA2E8BA3LL * ((a1[1] - *a1) >> 3);
  v9 = __CFADD__(v7, v6);
  v10 = v7 + v6;
  v11 = a2 - v5;
  if ( v9 )
  {
    v20 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v10 )
    {
      v24 = 0;
      v12 = 0;
      goto LABEL_7;
    }
    if ( v10 > 0x1745D1745D1745DLL )
      v10 = 0x1745D1745D1745DLL;
    v20 = 88 * v10;
  }
  v22 = a3;
  v21 = sub_22077B0(v20);
  v11 = a2 - v5;
  v12 = v21;
  a3 = v22;
  v24 = v20 + v21;
LABEL_7:
  v13 = v12 + v11;
  if ( v13 )
  {
    v14 = *a3;
    *(_DWORD *)(v13 + 4) = 0;
    *(_QWORD *)(v13 + 8) = 0;
    *(_DWORD *)v13 = v14;
    *(_QWORD *)(v13 + 16) = 0;
    *(_QWORD *)(v13 + 24) = 0;
    *(_QWORD *)(v13 + 32) = 0;
    *(_QWORD *)(v13 + 40) = 0;
    *(_QWORD *)(v13 + 48) = 0;
    *(_QWORD *)(v13 + 56) = 0;
    *(_QWORD *)(v13 + 64) = 0;
    *(_QWORD *)(v13 + 72) = 0;
    *(_QWORD *)(v13 + 80) = 0;
    sub_1371810((__int64 *)(v13 + 8), 0);
  }
  v15 = sub_13761C0(v5, a2, v12);
  v16 = a2;
  v17 = v5;
  v23 = sub_13761C0(v16, v4, v15 + 88);
  while ( v17 != v4 )
  {
    v18 = (__int64 *)(v17 + 8);
    v17 += 88;
    sub_13713C0(v18);
  }
  if ( v5 )
    j_j___libc_free_0(v5, a1[2] - v5);
  *a1 = v12;
  a1[1] = v23;
  a1[2] = v24;
  return v24;
}
