// Function: sub_3890D60
// Address: 0x3890d60
//
unsigned __int64 __fastcall sub_3890D60(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  _BYTE *v4; // rsi
  unsigned __int64 v5; // r12
  __int64 v6; // rdx
  _BYTE *v7; // rsi
  __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // r15
  unsigned int v13; // ebx
  size_t v14; // r13
  size_t v15; // r14
  size_t v16; // rdx
  unsigned int v17; // eax
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r13
  _QWORD *v22; // r13
  size_t v23; // rbx
  size_t v24; // r14
  size_t v25; // rdx
  int v26; // eax
  __int64 v27; // rbx
  unsigned int v28; // edi
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  size_t v33; // rbx
  size_t v34; // r14
  size_t v35; // rdx
  unsigned int v36; // edi
  __int64 v37; // rbx
  __int64 v38; // r13
  __int64 v39; // rax
  __int64 i; // rbx
  __int64 v41; // [rsp+0h] [rbp-60h]
  void *v42; // [rsp+10h] [rbp-50h]
  _QWORD *v43; // [rsp+18h] [rbp-48h]
  _QWORD *v44; // [rsp+20h] [rbp-40h]

  v3 = sub_22077B0(0xC8u);
  v4 = *(_BYTE **)(a2 + 32);
  v5 = v3;
  v6 = (__int64)&v4[*(_QWORD *)(a2 + 40)];
  *(_DWORD *)(v3 + 32) = *(_DWORD *)a2;
  *(_QWORD *)(v3 + 40) = *(_QWORD *)(a2 + 8);
  *(_DWORD *)(v3 + 48) = *(_DWORD *)(a2 + 16);
  *(_QWORD *)(v3 + 56) = *(_QWORD *)(a2 + 24);
  v41 = v3 + 80;
  *(_QWORD *)(v3 + 64) = v3 + 80;
  sub_3887850((__int64 *)(v3 + 64), v4, v6);
  v7 = *(_BYTE **)(a2 + 64);
  v8 = (__int64)&v7[*(_QWORD *)(a2 + 72)];
  *(_QWORD *)(v5 + 96) = v5 + 112;
  sub_3887850((__int64 *)(v5 + 96), v7, v8);
  v9 = *(_DWORD *)(a2 + 104);
  *(_DWORD *)(v5 + 136) = v9;
  if ( v9 > 0x40 )
    sub_16A4FD0(v5 + 128, (const void **)(a2 + 96));
  else
    *(_QWORD *)(v5 + 128) = *(_QWORD *)(a2 + 96);
  v43 = (_QWORD *)(v5 + 152);
  *(_BYTE *)(v5 + 140) = *(_BYTE *)(a2 + 108);
  v10 = (__int64 *)(a2 + 120);
  v42 = sub_16982C0();
  if ( *(void **)(a2 + 120) == v42 )
    sub_169C6E0(v43, (__int64)v10);
  else
    sub_16986C0((_QWORD *)(v5 + 152), v10);
  v11 = *(_QWORD *)(a2 + 144);
  *(_QWORD *)(v5 + 184) = 0;
  *(_QWORD *)(v5 + 192) = 0;
  *(_QWORD *)(v5 + 176) = v11;
  v12 = a1[2];
  v44 = a1 + 1;
  if ( !v12 )
  {
    v12 = (__int64)(a1 + 1);
    if ( v44 == (_QWORD *)a1[3] )
    {
      v22 = a1 + 1;
      goto LABEL_61;
    }
    goto LABEL_36;
  }
  v13 = *(_DWORD *)(v5 + 32);
  while ( 1 )
  {
    if ( v13 <= 1 )
    {
      LOBYTE(v18) = *(_DWORD *)(v5 + 48) < *(_DWORD *)(v12 + 48);
      goto LABEL_12;
    }
    v14 = *(_QWORD *)(v5 + 72);
    v15 = *(_QWORD *)(v12 + 72);
    v16 = v15;
    if ( v14 <= v15 )
      v16 = *(_QWORD *)(v5 + 72);
    if ( v16 )
    {
      v17 = memcmp(*(const void **)(v5 + 64), *(const void **)(v12 + 64), v16);
      if ( v17 )
      {
        v18 = v17 >> 31;
        goto LABEL_12;
      }
    }
    v21 = v14 - v15;
    if ( v21 >= 0x80000000LL )
    {
      v20 = *(_QWORD *)(v12 + 24);
LABEL_13:
      v19 = v20;
      LOBYTE(v18) = 0;
      goto LABEL_14;
    }
    if ( v21 > (__int64)0xFFFFFFFF7FFFFFFFLL )
    {
      LOBYTE(v18) = (int)v21 < 0;
LABEL_12:
      v19 = *(_QWORD *)(v12 + 16);
      v20 = *(_QWORD *)(v12 + 24);
      if ( (_BYTE)v18 )
        goto LABEL_14;
      goto LABEL_13;
    }
    v19 = *(_QWORD *)(v12 + 16);
    LOBYTE(v18) = 1;
LABEL_14:
    if ( !v19 )
      break;
    v12 = v19;
  }
  v22 = (_QWORD *)v12;
  if ( (_BYTE)v18 )
  {
    if ( a1[3] == v12 )
    {
LABEL_32:
      LOBYTE(v28) = 1;
      if ( v44 == v22 )
      {
LABEL_33:
        sub_220F040(v28, v5, v22, v44);
        ++a1[5];
        return v5;
      }
      if ( *(_DWORD *)(v5 + 32) <= 1u )
      {
        LOBYTE(v28) = *(_DWORD *)(v5 + 48) < *((_DWORD *)v22 + 12);
        goto LABEL_33;
      }
      v33 = *(_QWORD *)(v5 + 72);
      v34 = v22[9];
      v35 = v34;
      if ( v33 <= v34 )
        v35 = *(_QWORD *)(v5 + 72);
      if ( v35 )
      {
        v36 = memcmp(*(const void **)(v5 + 64), (const void *)v22[8], v35);
        if ( v36 )
        {
LABEL_58:
          v28 = v36 >> 31;
          goto LABEL_33;
        }
      }
      v37 = v33 - v34;
      LOBYTE(v28) = 0;
      if ( v37 > 0x7FFFFFFF )
        goto LABEL_33;
      if ( v37 >= (__int64)0xFFFFFFFF80000000LL )
      {
        v36 = v37;
        goto LABEL_58;
      }
LABEL_61:
      LOBYTE(v28) = 1;
      goto LABEL_33;
    }
LABEL_36:
    v22 = (_QWORD *)v12;
    v12 = sub_220EF80(v12);
    if ( *(_DWORD *)(v12 + 32) <= 1u )
      goto LABEL_37;
LABEL_23:
    v23 = *(_QWORD *)(v12 + 72);
    v24 = *(_QWORD *)(v5 + 72);
    v25 = v24;
    if ( v23 <= v24 )
      v25 = *(_QWORD *)(v12 + 72);
    if ( !v25 || (v26 = memcmp(*(const void **)(v12 + 64), *(const void **)(v5 + 64), v25)) == 0 )
    {
      v27 = v23 - v24;
      if ( v27 > 0x7FFFFFFF )
        goto LABEL_38;
      if ( v27 < (__int64)0xFFFFFFFF80000000LL )
        goto LABEL_31;
      v26 = v27;
    }
    if ( v26 >= 0 )
      goto LABEL_38;
LABEL_31:
    if ( !v22 )
    {
      v12 = 0;
      goto LABEL_38;
    }
    goto LABEL_32;
  }
  if ( *(_DWORD *)(v12 + 32) > 1u )
    goto LABEL_23;
LABEL_37:
  if ( *(_DWORD *)(v12 + 48) < *(_DWORD *)(v5 + 48) )
    goto LABEL_31;
LABEL_38:
  if ( v42 == *(void **)(v5 + 152) )
  {
    v38 = *(_QWORD *)(v5 + 160);
    if ( v38 )
    {
      v39 = 32LL * *(_QWORD *)(v38 - 8);
      for ( i = v38 + v39; v38 != i; sub_127D120((_QWORD *)(i + 8)) )
        i -= 32;
      j_j_j___libc_free_0_0(v38 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)v43);
  }
  if ( *(_DWORD *)(v5 + 136) > 0x40u )
  {
    v30 = *(_QWORD *)(v5 + 128);
    if ( v30 )
      j_j___libc_free_0_0(v30);
  }
  v31 = *(_QWORD *)(v5 + 96);
  if ( v5 + 112 != v31 )
    j_j___libc_free_0(v31);
  v32 = *(_QWORD *)(v5 + 64);
  if ( v41 != v32 )
    j_j___libc_free_0(v32);
  j_j___libc_free_0(v5);
  return v12;
}
