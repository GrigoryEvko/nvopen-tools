// Function: sub_2BFA260
// Address: 0x2bfa260
//
__int64 __fastcall sub_2BFA260(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // r12
  const void *v8; // rbx
  __int64 v9; // r15
  __int64 v10; // rax
  _BYTE *v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r12
  _BYTE *v17; // rsi
  __int64 v18; // rdx
  __int64 *v19; // rdi
  __int64 v20; // rax
  unsigned __int64 v21; // rcx
  __int64 v22; // rdx
  _QWORD *v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // r13
  __int64 v26; // rbx
  __int64 v27; // rdi
  _BYTE *v29; // rdi
  _BYTE *v31; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v32; // [rsp+18h] [rbp-B8h]
  _BYTE dest[16]; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v34[2]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD v35[2]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v36[2]; // [rsp+50h] [rbp-80h] BYREF
  __int64 v37; // [rsp+60h] [rbp-70h] BYREF
  __int64 *v38; // [rsp+70h] [rbp-60h] BYREF
  __int16 v39; // [rsp+90h] [rbp-40h]

  v7 = *(unsigned int *)(a1 + 88);
  v31 = dest;
  v8 = *(const void **)(a1 + 80);
  v32 = 0x200000000LL;
  v9 = 8 * v7;
  if ( v7 > 2 )
  {
    sub_C8D5F0((__int64)&v31, dest, v7, 8u, a5, a6);
    v29 = &v31[8 * (unsigned int)v32];
  }
  else
  {
    if ( !v9 )
      goto LABEL_3;
    v29 = dest;
  }
  memcpy(v29, v8, 8 * v7);
  LODWORD(v9) = v32;
LABEL_3:
  LODWORD(v32) = v9 + v7;
  v10 = sub_2BF9BD0(a1);
  v11 = *(_BYTE **)(a1 + 16);
  v12 = *(_QWORD *)(a1 + 24);
  v13 = v10;
  v34[0] = (__int64)v35;
  sub_2BEF590(v34, v11, (__int64)&v11[v12]);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v34[1]) <= 5 )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490((unsigned __int64 *)v34, ".split", 6u);
  v38 = v34;
  v39 = 260;
  v16 = sub_22077B0(0x80u);
  if ( v16 )
  {
    sub_CA0F50(v36, (void **)&v38);
    v17 = (_BYTE *)v36[0];
    *(_BYTE *)(v16 + 8) = 1;
    v18 = v36[1];
    *(_QWORD *)v16 = &unk_4A23970;
    *(_QWORD *)(v16 + 16) = v16 + 32;
    sub_2BEF590((__int64 *)(v16 + 16), v17, (__int64)&v17[v18]);
    v19 = (__int64 *)v36[0];
    *(_QWORD *)(v16 + 56) = v16 + 72;
    *(_QWORD *)(v16 + 64) = 0x100000000LL;
    *(_QWORD *)(v16 + 88) = 0x100000000LL;
    *(_QWORD *)(v16 + 48) = 0;
    *(_QWORD *)(v16 + 80) = v16 + 96;
    *(_QWORD *)(v16 + 104) = 0;
    if ( v19 != &v37 )
      j_j___libc_free_0((unsigned __int64)v19);
    *(_QWORD *)v16 = &unk_4A23A00;
    *(_QWORD *)(v16 + 120) = v16 + 112;
    *(_QWORD *)(v16 + 112) = (v16 + 112) | 4;
  }
  v20 = *(unsigned int *)(v13 + 600);
  v21 = *(unsigned int *)(v13 + 604);
  if ( v20 + 1 > v21 )
  {
    sub_C8D5F0(v13 + 592, (const void *)(v13 + 608), v20 + 1, 8u, v14, v15);
    v20 = *(unsigned int *)(v13 + 600);
  }
  v22 = *(_QWORD *)(v13 + 592);
  *(_QWORD *)(v22 + 8 * v20) = v16;
  v23 = (_QWORD *)v34[0];
  ++*(_DWORD *)(v13 + 600);
  if ( v23 != v35 )
    j_j___libc_free_0((unsigned __int64)v23);
  v24 = a1;
  v25 = a1 + 112;
  sub_2BEFE80(v16, v24, v22, v21, v14, v15);
  v26 = a2;
  while ( v25 != v26 )
  {
    v27 = v26;
    v26 = *(_QWORD *)(v26 + 8);
    sub_2C19EE0(v27 - 24, v16, v16 + 112);
  }
  if ( v31 != dest )
    _libc_free((unsigned __int64)v31);
  return v16;
}
