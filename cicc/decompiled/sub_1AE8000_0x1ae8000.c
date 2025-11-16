// Function: sub_1AE8000
// Address: 0x1ae8000
//
void __fastcall sub_1AE8000(__int64 *a1, __int64 a2, const void *a3, __int64 a4)
{
  const void *v4; // r9
  size_t v6; // r12
  __int64 v8; // r14
  int v9; // edx
  int v10; // r12d
  __int64 v11; // r14
  _QWORD *v12; // r15
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 *v16; // r12
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 *v19; // rax
  __int64 v20; // rsi
  unsigned __int64 v21; // rcx
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 *v24; // rdx
  __int64 v25; // rsi
  unsigned __int64 v26; // rcx
  __int64 v27; // rcx
  _BYTE *v28; // rdi
  _BYTE *v30; // [rsp+10h] [rbp-80h] BYREF
  __int64 v31; // [rsp+18h] [rbp-78h]
  _BYTE dest[112]; // [rsp+20h] [rbp-70h] BYREF

  v4 = a3;
  v6 = 8 * a4;
  v8 = (8 * a4) >> 3;
  v30 = dest;
  v31 = 0x800000000LL;
  if ( (unsigned __int64)(8 * a4) > 0x40 )
  {
    sub_16CD150((__int64)&v30, dest, (8 * a4) >> 3, 8, (int)&v30, (int)a3);
    v4 = a3;
    v28 = &v30[8 * (unsigned int)v31];
  }
  else
  {
    if ( !v6 )
      goto LABEL_3;
    v28 = dest;
  }
  memcpy(v28, v4, v6);
  LODWORD(v6) = v31;
LABEL_3:
  v9 = *(_DWORD *)(a2 + 20);
  v10 = v8 + v6;
  v11 = *a1;
  LODWORD(v31) = v10;
  v12 = *(_QWORD **)(*(_QWORD *)(a2 + 24 * (2LL - (v9 & 0xFFFFFFF))) + 24LL);
  if ( v10 )
  {
    v13 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v13 + 16) )
      BUG();
    v12 = (_QWORD *)sub_15C46E0(v12, (__int64)&v30, *(_DWORD *)(v13 + 36) == 38);
  }
  v14 = *(_QWORD *)(v11 + 8);
  if ( (*(_BYTE *)(v14 + 23) & 0x40) != 0 )
    v15 = *(__int64 **)(v14 - 8);
  else
    v15 = (__int64 *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
  v16 = **(__int64 ***)v11;
  v17 = sub_1624210(*v15);
  v18 = sub_1628DA0(v16, (__int64)v17);
  v19 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( *v19 )
  {
    v20 = v19[1];
    v21 = v19[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v21 = v20;
    if ( v20 )
      *(_QWORD *)(v20 + 16) = *(_QWORD *)(v20 + 16) & 3LL | v21;
  }
  *v19 = v18;
  if ( v18 )
  {
    v22 = *(_QWORD *)(v18 + 8);
    v19[1] = v22;
    if ( v22 )
      *(_QWORD *)(v22 + 16) = (unsigned __int64)(v19 + 1) | *(_QWORD *)(v22 + 16) & 3LL;
    v19[2] = (v18 + 8) | v19[2] & 3;
    *(_QWORD *)(v18 + 8) = v19;
  }
  v23 = sub_1628DA0(*(__int64 **)(v11 + 16), (__int64)v12);
  v24 = (__int64 *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( *v24 )
  {
    v25 = v24[1];
    v26 = v24[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v26 = v25;
    if ( v25 )
      *(_QWORD *)(v25 + 16) = *(_QWORD *)(v25 + 16) & 3LL | v26;
  }
  *v24 = v23;
  if ( v23 )
  {
    v27 = *(_QWORD *)(v23 + 8);
    v24[1] = v27;
    if ( v27 )
      *(_QWORD *)(v27 + 16) = (unsigned __int64)(v24 + 1) | *(_QWORD *)(v27 + 16) & 3LL;
    v24[2] = (v23 + 8) | v24[2] & 3;
    *(_QWORD *)(v23 + 8) = v24;
  }
  if ( v30 != dest )
    _libc_free((unsigned __int64)v30);
}
