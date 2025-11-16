// Function: sub_3890800
// Address: 0x3890800
//
unsigned __int64 __fastcall sub_3890800(_QWORD *a1, __int64 a2)
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
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  int v14; // esi
  __int64 v15; // rax
  __int64 v16; // r15
  unsigned int v17; // ebx
  size_t v18; // r13
  size_t v19; // r14
  size_t v20; // rdx
  unsigned int v21; // eax
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r13
  _QWORD *v26; // r13
  size_t v27; // rbx
  size_t v28; // r14
  size_t v29; // rdx
  int v30; // eax
  __int64 v31; // rbx
  unsigned int v32; // edi
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rdi
  size_t v38; // rbx
  size_t v39; // r14
  size_t v40; // rdx
  unsigned int v41; // edi
  __int64 v42; // rbx
  __int64 v43; // r13
  __int64 v44; // rax
  __int64 i; // rbx
  __int64 v46; // [rsp+0h] [rbp-60h]
  void *v47; // [rsp+10h] [rbp-50h]
  _QWORD *v48; // [rsp+18h] [rbp-48h]
  _QWORD *v49; // [rsp+20h] [rbp-40h]

  v3 = sub_22077B0(0xF0u);
  v4 = *(_BYTE **)(a2 + 32);
  v5 = v3;
  v6 = (__int64)&v4[*(_QWORD *)(a2 + 40)];
  *(_DWORD *)(v3 + 32) = *(_DWORD *)a2;
  *(_QWORD *)(v3 + 40) = *(_QWORD *)(a2 + 8);
  *(_DWORD *)(v3 + 48) = *(_DWORD *)(a2 + 16);
  *(_QWORD *)(v3 + 56) = *(_QWORD *)(a2 + 24);
  v46 = v3 + 80;
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
  v48 = (_QWORD *)(v5 + 152);
  *(_BYTE *)(v5 + 140) = *(_BYTE *)(a2 + 108);
  v10 = (__int64 *)(a2 + 120);
  v47 = sub_16982C0();
  if ( *(void **)(a2 + 120) == v47 )
    sub_169C6E0(v48, (__int64)v10);
  else
    sub_16986C0((_QWORD *)(v5 + 152), v10);
  v11 = *(_QWORD *)(a2 + 144);
  v12 = v5 + 200;
  *(_QWORD *)(v5 + 184) = 0;
  *(_QWORD *)(v5 + 176) = v11;
  v13 = *(_QWORD *)(a2 + 176);
  if ( v13 )
  {
    v14 = *(_DWORD *)(a2 + 168);
    *(_QWORD *)(v5 + 208) = v13;
    *(_DWORD *)(v5 + 200) = v14;
    *(_QWORD *)(v5 + 216) = *(_QWORD *)(a2 + 184);
    *(_QWORD *)(v5 + 224) = *(_QWORD *)(a2 + 192);
    *(_QWORD *)(v13 + 8) = v12;
    v15 = *(_QWORD *)(a2 + 200);
    *(_QWORD *)(a2 + 176) = 0;
    *(_QWORD *)(v5 + 232) = v15;
    *(_QWORD *)(a2 + 184) = a2 + 168;
    *(_QWORD *)(a2 + 192) = a2 + 168;
    *(_QWORD *)(a2 + 200) = 0;
  }
  else
  {
    *(_DWORD *)(v5 + 200) = 0;
    *(_QWORD *)(v5 + 208) = 0;
    *(_QWORD *)(v5 + 216) = v12;
    *(_QWORD *)(v5 + 224) = v12;
    *(_QWORD *)(v5 + 232) = 0;
  }
  v16 = a1[2];
  v49 = a1 + 1;
  if ( !v16 )
  {
    v16 = (__int64)(a1 + 1);
    if ( v49 == (_QWORD *)a1[3] )
    {
      v26 = a1 + 1;
      goto LABEL_66;
    }
    goto LABEL_38;
  }
  v17 = *(_DWORD *)(v5 + 32);
  while ( 1 )
  {
    if ( v17 <= 1 )
    {
      LOBYTE(v22) = *(_DWORD *)(v5 + 48) < *(_DWORD *)(v16 + 48);
      goto LABEL_14;
    }
    v18 = *(_QWORD *)(v5 + 72);
    v19 = *(_QWORD *)(v16 + 72);
    v20 = v19;
    if ( v18 <= v19 )
      v20 = *(_QWORD *)(v5 + 72);
    if ( v20 )
    {
      v21 = memcmp(*(const void **)(v5 + 64), *(const void **)(v16 + 64), v20);
      if ( v21 )
      {
        v22 = v21 >> 31;
        goto LABEL_14;
      }
    }
    v25 = v18 - v19;
    if ( v25 >= 0x80000000LL )
    {
      v24 = *(_QWORD *)(v16 + 24);
LABEL_15:
      v23 = v24;
      LOBYTE(v22) = 0;
      goto LABEL_16;
    }
    if ( v25 > (__int64)0xFFFFFFFF7FFFFFFFLL )
    {
      LOBYTE(v22) = (int)v25 < 0;
LABEL_14:
      v23 = *(_QWORD *)(v16 + 16);
      v24 = *(_QWORD *)(v16 + 24);
      if ( (_BYTE)v22 )
        goto LABEL_16;
      goto LABEL_15;
    }
    v23 = *(_QWORD *)(v16 + 16);
    LOBYTE(v22) = 1;
LABEL_16:
    if ( !v23 )
      break;
    v16 = v23;
  }
  v26 = (_QWORD *)v16;
  if ( (_BYTE)v22 )
  {
    if ( a1[3] == v16 )
    {
LABEL_34:
      LOBYTE(v32) = 1;
      if ( v49 == v26 )
      {
LABEL_35:
        sub_220F040(v32, v5, v26, v49);
        ++a1[5];
        return v5;
      }
      if ( *(_DWORD *)(v5 + 32) <= 1u )
      {
        LOBYTE(v32) = *(_DWORD *)(v5 + 48) < *((_DWORD *)v26 + 12);
        goto LABEL_35;
      }
      v38 = *(_QWORD *)(v5 + 72);
      v39 = v26[9];
      v40 = v39;
      if ( v38 <= v39 )
        v40 = *(_QWORD *)(v5 + 72);
      if ( v40 )
      {
        v41 = memcmp(*(const void **)(v5 + 64), (const void *)v26[8], v40);
        if ( v41 )
        {
LABEL_63:
          v32 = v41 >> 31;
          goto LABEL_35;
        }
      }
      v42 = v38 - v39;
      LOBYTE(v32) = 0;
      if ( v42 > 0x7FFFFFFF )
        goto LABEL_35;
      if ( v42 >= (__int64)0xFFFFFFFF80000000LL )
      {
        v41 = v42;
        goto LABEL_63;
      }
LABEL_66:
      LOBYTE(v32) = 1;
      goto LABEL_35;
    }
LABEL_38:
    v26 = (_QWORD *)v16;
    v16 = sub_220EF80(v16);
    if ( *(_DWORD *)(v16 + 32) <= 1u )
      goto LABEL_39;
LABEL_25:
    v27 = *(_QWORD *)(v16 + 72);
    v28 = *(_QWORD *)(v5 + 72);
    v29 = v28;
    if ( v27 <= v28 )
      v29 = *(_QWORD *)(v16 + 72);
    if ( !v29 || (v30 = memcmp(*(const void **)(v16 + 64), *(const void **)(v5 + 64), v29)) == 0 )
    {
      v31 = v27 - v28;
      if ( v31 > 0x7FFFFFFF )
        goto LABEL_40;
      if ( v31 < (__int64)0xFFFFFFFF80000000LL )
        goto LABEL_33;
      v30 = v31;
    }
    if ( v30 >= 0 )
      goto LABEL_40;
LABEL_33:
    if ( !v26 )
    {
      v16 = 0;
      goto LABEL_40;
    }
    goto LABEL_34;
  }
  if ( *(_DWORD *)(v16 + 32) > 1u )
    goto LABEL_25;
LABEL_39:
  if ( *(_DWORD *)(v16 + 48) < *(_DWORD *)(v5 + 48) )
    goto LABEL_33;
LABEL_40:
  sub_3889CD0(*(_QWORD *)(v5 + 208));
  v34 = *(_QWORD *)(v5 + 184);
  if ( v34 )
    j_j___libc_free_0_0(v34);
  if ( v47 == *(void **)(v5 + 152) )
  {
    v43 = *(_QWORD *)(v5 + 160);
    if ( v43 )
    {
      v44 = 32LL * *(_QWORD *)(v43 - 8);
      for ( i = v43 + v44; v43 != i; sub_127D120((_QWORD *)(i + 8)) )
        i -= 32;
      j_j_j___libc_free_0_0(v43 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)v48);
  }
  if ( *(_DWORD *)(v5 + 136) > 0x40u )
  {
    v35 = *(_QWORD *)(v5 + 128);
    if ( v35 )
      j_j___libc_free_0_0(v35);
  }
  v36 = *(_QWORD *)(v5 + 96);
  if ( v5 + 112 != v36 )
    j_j___libc_free_0(v36);
  v37 = *(_QWORD *)(v5 + 64);
  if ( v46 != v37 )
    j_j___libc_free_0(v37);
  j_j___libc_free_0(v5);
  return v16;
}
