// Function: sub_1087810
// Address: 0x1087810
//
void __fastcall sub_1087810(__int64 a1)
{
  __int64 *v1; // r12
  __int64 *v3; // r13
  char *v4; // r15
  char *v5; // r14
  __int64 v6; // rdx
  signed __int64 v7; // r9
  __int64 v8; // rax
  __int64 v9; // rdi
  bool v10; // cf
  unsigned __int64 v11; // rax
  __int64 v12; // r8
  char *v13; // r10
  unsigned __int64 v14; // rax
  char *v15; // r15
  __int64 v16; // r12
  unsigned int v17; // eax
  __int64 v18; // rdi
  unsigned int v19; // eax
  __int64 v20; // rdi
  unsigned int v21; // eax
  __int64 v22; // rdi
  unsigned int v23; // eax
  __int64 v24; // rdi
  unsigned int v25; // eax
  __int64 v26; // rdi
  unsigned int v27; // eax
  __int64 v28; // rdi
  __int16 v29; // ax
  __int64 v30; // rdi
  __int16 v31; // ax
  __int64 v32; // rdi
  unsigned int v33; // eax
  __int64 v34; // rdi
  char *v35; // rax
  char *i; // rdi
  __int64 v37; // rcx
  __int64 v38; // rdx
  char *v39; // rsi
  char *v40; // rax
  __int64 v41; // r14
  __int64 v42; // rax
  __int64 v43; // [rsp+8h] [rbp-58h]
  signed __int64 v44; // [rsp+8h] [rbp-58h]
  signed __int64 v45; // [rsp+10h] [rbp-50h]
  __int64 v46; // [rsp+10h] [rbp-50h]
  char *v47; // [rsp+10h] [rbp-50h]
  __int64 v48; // [rsp+10h] [rbp-50h]
  char *src; // [rsp+18h] [rbp-48h]
  unsigned __int8 v50[52]; // [rsp+2Ch] [rbp-34h] BYREF

  v1 = *(__int64 **)(a1 + 56);
  if ( *(__int64 **)(a1 + 48) == v1 )
    return;
  v3 = *(__int64 **)(a1 + 48);
  v4 = 0;
  v5 = 0;
  src = 0;
  do
  {
    while ( 1 )
    {
      v6 = *v3;
      if ( v5 != v4 )
        break;
      v7 = v5 - src;
      v8 = (v5 - src) >> 3;
      if ( v8 == 0xFFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"vector::_M_realloc_insert");
      v9 = 1;
      if ( v8 )
        v9 = (v5 - src) >> 3;
      v10 = __CFADD__(v9, v8);
      v11 = v9 + v8;
      if ( v10 )
      {
        v41 = 0x7FFFFFFFFFFFFFF8LL;
      }
      else
      {
        if ( !v11 )
        {
          v12 = 0;
          v13 = 0;
          goto LABEL_13;
        }
        if ( v11 > 0xFFFFFFFFFFFFFFFLL )
          v11 = 0xFFFFFFFFFFFFFFFLL;
        v41 = 8 * v11;
      }
      v44 = v7;
      v48 = *v3;
      v42 = sub_22077B0(v41);
      v6 = v48;
      v7 = v44;
      v13 = (char *)v42;
      v12 = v42 + v41;
LABEL_13:
      if ( &v13[v7] )
        *(_QWORD *)&v13[v7] = v6;
      v5 = &v13[v7 + 8];
      if ( v7 > 0 )
      {
        v46 = v12;
        v35 = (char *)memmove(v13, src, v7);
        v12 = v46;
        v13 = v35;
LABEL_48:
        v43 = v12;
        v47 = v13;
        j_j___libc_free_0(src, v4 - src);
        v12 = v43;
        v13 = v47;
        goto LABEL_17;
      }
      if ( src )
        goto LABEL_48;
LABEL_17:
      ++v3;
      src = v13;
      v4 = (char *)v12;
      if ( v1 == v3 )
        goto LABEL_18;
    }
    if ( v5 )
      *(_QWORD *)v5 = v6;
    ++v3;
    v5 += 8;
  }
  while ( v1 != v3 );
LABEL_18:
  v45 = v4 - src;
  if ( v5 != src )
  {
    _BitScanReverse64(&v14, (v5 - src) >> 3);
    sub_1084A90(src, v5, 2LL * (int)(63 - (v14 ^ 0x3F)));
    if ( v5 - src > 128 )
    {
      sub_10849D0(src, src + 128);
      for ( i = src + 128; v5 != i; *(_QWORD *)v39 = v37 )
      {
        v37 = *(_QWORD *)i;
        v38 = *((_QWORD *)i - 1);
        v39 = i;
        v40 = i - 8;
        if ( *(_DWORD *)(v38 + 72) > *(_DWORD *)(*(_QWORD *)i + 72LL) )
        {
          do
          {
            *((_QWORD *)v40 + 1) = v38;
            v39 = v40;
            v38 = *((_QWORD *)v40 - 1);
            v40 -= 8;
          }
          while ( *(_DWORD *)(v37 + 72) < *(_DWORD *)(v38 + 72) );
        }
        i += 8;
      }
    }
    else
    {
      sub_10849D0(src, v5);
    }
    v15 = src;
    do
    {
      v16 = *(_QWORD *)v15;
      if ( *(_DWORD *)(*(_QWORD *)v15 + 72LL) != -1 )
      {
        if ( *(_QWORD *)(v16 + 104) - *(_QWORD *)(v16 + 96) > 0x17FFD0u )
          *(_DWORD *)(v16 + 36) |= 0x1000000u;
        sub_CB6200(*(_QWORD *)(a1 + 8), (unsigned __int8 *)v16, 8u);
        v17 = *(_DWORD *)(v16 + 8);
        v18 = *(_QWORD *)(a1 + 8);
        if ( *(_DWORD *)(a1 + 16) != 1 )
          v17 = _byteswap_ulong(v17);
        *(_DWORD *)v50 = v17;
        sub_CB6200(v18, v50, 4u);
        v19 = *(_DWORD *)(v16 + 12);
        v20 = *(_QWORD *)(a1 + 8);
        if ( *(_DWORD *)(a1 + 16) != 1 )
          v19 = _byteswap_ulong(v19);
        *(_DWORD *)v50 = v19;
        sub_CB6200(v20, v50, 4u);
        v21 = *(_DWORD *)(v16 + 16);
        v22 = *(_QWORD *)(a1 + 8);
        if ( *(_DWORD *)(a1 + 16) != 1 )
          v21 = _byteswap_ulong(v21);
        *(_DWORD *)v50 = v21;
        sub_CB6200(v22, v50, 4u);
        v23 = *(_DWORD *)(v16 + 20);
        v24 = *(_QWORD *)(a1 + 8);
        if ( *(_DWORD *)(a1 + 16) != 1 )
          v23 = _byteswap_ulong(v23);
        *(_DWORD *)v50 = v23;
        sub_CB6200(v24, v50, 4u);
        v25 = *(_DWORD *)(v16 + 24);
        v26 = *(_QWORD *)(a1 + 8);
        if ( *(_DWORD *)(a1 + 16) != 1 )
          v25 = _byteswap_ulong(v25);
        *(_DWORD *)v50 = v25;
        sub_CB6200(v26, v50, 4u);
        v27 = *(_DWORD *)(v16 + 28);
        v28 = *(_QWORD *)(a1 + 8);
        if ( *(_DWORD *)(a1 + 16) != 1 )
          v27 = _byteswap_ulong(v27);
        *(_DWORD *)v50 = v27;
        sub_CB6200(v28, v50, 4u);
        v29 = *(_WORD *)(v16 + 32);
        v30 = *(_QWORD *)(a1 + 8);
        if ( *(_DWORD *)(a1 + 16) != 1 )
          v29 = __ROL2__(v29, 8);
        *(_WORD *)v50 = v29;
        sub_CB6200(v30, v50, 2u);
        v31 = *(_WORD *)(v16 + 34);
        v32 = *(_QWORD *)(a1 + 8);
        if ( *(_DWORD *)(a1 + 16) != 1 )
          v31 = __ROL2__(v31, 8);
        *(_WORD *)v50 = v31;
        sub_CB6200(v32, v50, 2u);
        v33 = *(_DWORD *)(v16 + 36);
        v34 = *(_QWORD *)(a1 + 8);
        if ( *(_DWORD *)(a1 + 16) != 1 )
          v33 = _byteswap_ulong(v33);
        *(_DWORD *)v50 = v33;
        sub_CB6200(v34, v50, 4u);
      }
      v15 += 8;
    }
    while ( v5 != v15 );
  }
  if ( src )
    j_j___libc_free_0(src, v45);
}
