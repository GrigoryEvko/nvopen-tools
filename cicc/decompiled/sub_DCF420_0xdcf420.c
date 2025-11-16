// Function: sub_DCF420
// Address: 0xdcf420
//
__int64 __fastcall sub_DCF420(__int64 *a1, __int64 a2)
{
  unsigned int v2; // r14d
  __int16 v3; // ax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rbx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rdi
  unsigned int v19; // ebx
  unsigned __int64 v20; // rax
  int v21; // r15d
  unsigned __int64 v22; // r15
  unsigned int v23; // ebx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rbx
  __int64 v28; // r8
  __int64 v29; // rax
  __int64 v30; // rbx
  const void *v31; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v32; // [rsp+18h] [rbp-88h]
  __int64 v33; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v34; // [rsp+28h] [rbp-78h]
  const void *v35; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v36; // [rsp+38h] [rbp-68h]
  __int64 v37; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v38; // [rsp+48h] [rbp-58h]
  const void *v39; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v40; // [rsp+58h] [rbp-48h]
  __int64 v41; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v42; // [rsp+68h] [rbp-38h]

  v2 = 0;
  if ( *(_QWORD *)(a2 + 40) != 2 )
    return v2;
  v3 = *(_WORD *)(a2 + 28);
  if ( (v3 & 1) == 0 )
  {
    v15 = sub_DCF3A0(a1, *(char **)(a2 + 48), 1);
    if ( !*(_WORD *)(v15 + 24) )
    {
      v16 = sub_D33D80((_QWORD *)a2, (__int64)a1, v12, v13, v14);
      v17 = sub_DBB9F0((__int64)a1, v16, 1u, 0);
      v40 = *(_DWORD *)(v17 + 8);
      if ( v40 > 0x40 )
        sub_C43780((__int64)&v39, (const void **)v17);
      else
        v39 = *(const void **)v17;
      v42 = *(_DWORD *)(v17 + 24);
      if ( v42 > 0x40 )
        sub_C43780((__int64)&v41, (const void **)(v17 + 16));
      else
        v41 = *(_QWORD *)(v17 + 16);
      v18 = *(_QWORD *)(v15 + 32);
      v19 = *(_DWORD *)(v18 + 32);
      if ( v19 > 0x40 )
      {
        v21 = sub_C444A0(v18 + 24);
      }
      else
      {
        v20 = *(_QWORD *)(v18 + 24);
        v21 = *(_DWORD *)(v18 + 32);
        if ( v20 )
        {
          _BitScanReverse64(&v22, v20);
          v21 = v19 - 64 + (v22 ^ 0x3F);
        }
      }
      v23 = sub_AB1D50((__int64)&v39) + v19 - v21;
      v24 = sub_D95540(**(_QWORD **)(a2 + 32));
      v2 = v23 <= (unsigned __int64)sub_D97050((__int64)a1, v24);
      if ( v42 > 0x40 && v41 )
        j_j___libc_free_0_0(v41);
      if ( v40 > 0x40 && v39 )
        j_j___libc_free_0_0(v39);
    }
    v3 = *(_WORD *)(a2 + 28);
    if ( (v3 & 4) != 0 )
      goto LABEL_4;
    goto LABEL_49;
  }
  if ( (v3 & 4) == 0 )
  {
LABEL_49:
    v27 = sub_DBB9F0((__int64)a1, a2, 1u, 0);
    v32 = *(_DWORD *)(v27 + 8);
    if ( v32 > 0x40 )
      sub_C43780((__int64)&v31, (const void **)v27);
    else
      v31 = *(const void **)v27;
    v34 = *(_DWORD *)(v27 + 24);
    if ( v34 > 0x40 )
      sub_C43780((__int64)&v33, (const void **)(v27 + 16));
    else
      v33 = *(_QWORD *)(v27 + 16);
    v29 = sub_D33D80((_QWORD *)a2, (__int64)a1, v25, v26, v28);
    v30 = sub_DBB9F0((__int64)a1, v29, 1u, 0);
    v36 = *(_DWORD *)(v30 + 8);
    if ( v36 > 0x40 )
      sub_C43780((__int64)&v35, (const void **)v30);
    else
      v35 = *(const void **)v30;
    v38 = *(_DWORD *)(v30 + 24);
    if ( v38 > 0x40 )
      sub_C43780((__int64)&v37, (const void **)(v30 + 16));
    else
      v37 = *(_QWORD *)(v30 + 16);
    sub_AB28E0((__int64)&v39, 0xDu, (__int64)&v35, 2);
    if ( (unsigned __int8)sub_AB1BB0((__int64)&v39, (__int64)&v31) )
      v2 |= 4u;
    if ( v42 > 0x40 && v41 )
      j_j___libc_free_0_0(v41);
    if ( v40 > 0x40 && v39 )
      j_j___libc_free_0_0(v39);
    if ( v38 > 0x40 && v37 )
      j_j___libc_free_0_0(v37);
    if ( v36 > 0x40 && v35 )
      j_j___libc_free_0_0(v35);
    if ( v34 > 0x40 && v33 )
      j_j___libc_free_0_0(v33);
    if ( v32 > 0x40 && v31 )
      j_j___libc_free_0_0(v31);
    v3 = *(_WORD *)(a2 + 28);
  }
LABEL_4:
  if ( (v3 & 2) == 0 )
  {
    v8 = sub_DBB9F0((__int64)a1, a2, 0, 0);
    v32 = *(_DWORD *)(v8 + 8);
    if ( v32 > 0x40 )
      sub_C43780((__int64)&v31, (const void **)v8);
    else
      v31 = *(const void **)v8;
    v34 = *(_DWORD *)(v8 + 24);
    if ( v34 > 0x40 )
      sub_C43780((__int64)&v33, (const void **)(v8 + 16));
    else
      v33 = *(_QWORD *)(v8 + 16);
    v10 = sub_D33D80((_QWORD *)a2, (__int64)a1, v6, v7, v9);
    v11 = sub_DBB9F0((__int64)a1, v10, 0, 0);
    v36 = *(_DWORD *)(v11 + 8);
    if ( v36 > 0x40 )
      sub_C43780((__int64)&v35, (const void **)v11);
    else
      v35 = *(const void **)v11;
    v38 = *(_DWORD *)(v11 + 24);
    if ( v38 > 0x40 )
      sub_C43780((__int64)&v37, (const void **)(v11 + 16));
    else
      v37 = *(_QWORD *)(v11 + 16);
    sub_AB28E0((__int64)&v39, 0xDu, (__int64)&v35, 1);
    if ( (unsigned __int8)sub_AB1BB0((__int64)&v39, (__int64)&v31) )
      v2 |= 2u;
    if ( v42 > 0x40 && v41 )
      j_j___libc_free_0_0(v41);
    if ( v40 > 0x40 && v39 )
      j_j___libc_free_0_0(v39);
    if ( v38 > 0x40 && v37 )
      j_j___libc_free_0_0(v37);
    if ( v36 > 0x40 && v35 )
      j_j___libc_free_0_0(v35);
    if ( v34 > 0x40 && v33 )
      j_j___libc_free_0_0(v33);
    if ( v32 > 0x40 && v31 )
      j_j___libc_free_0_0(v31);
  }
  return v2;
}
