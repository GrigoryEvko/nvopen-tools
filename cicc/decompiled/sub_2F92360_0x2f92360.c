// Function: sub_2F92360
// Address: 0x2f92360
//
__int64 __fastcall sub_2F92360(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r15
  _QWORD *v9; // rbx
  __int64 v10; // rcx
  _QWORD *v11; // r15
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 v14; // r14
  __int64 v15; // rdi
  int v16; // eax
  __int64 v17; // rdi
  int v18; // eax
  int v19; // eax
  unsigned int v20; // ecx
  __int64 v21; // rdx
  __int64 v22; // rdx
  _QWORD *v23; // rax
  _QWORD *i; // rdx
  _QWORD **v25; // r15
  _QWORD **v26; // rax
  _QWORD **v27; // r12
  _QWORD *v28; // rbx
  _QWORD **v29; // r13
  unsigned __int64 v30; // rdi
  _QWORD *v32; // rax
  __int64 v33; // rdx
  _QWORD *j; // rdx
  unsigned int v35; // eax
  unsigned int v36; // ebx
  char v37; // al
  __int64 v38; // rdi
  __int64 v39; // rax
  _QWORD *v40; // rax
  __int64 v41; // rdx
  _QWORD *v42; // rdx
  _QWORD *v43; // [rsp+8h] [rbp-48h]
  __int64 v44; // [rsp+8h] [rbp-48h]
  __int64 v45; // [rsp+8h] [rbp-48h]
  __int64 v46; // [rsp+10h] [rbp-40h] BYREF
  __int64 v47; // [rsp+18h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 80);
  v8 = 32LL * *(unsigned int *)(a2 + 88);
  v9 = (_QWORD *)(v7 + 8);
  v10 = v7 + v8;
  v43 = (_QWORD *)(v7 + v8);
  if ( v7 != v7 + v8 )
  {
    while ( 1 )
    {
      v11 = (_QWORD *)*v9;
      if ( (_QWORD *)*v9 != v9 )
        break;
LABEL_19:
      if ( v43 == v9 + 3 )
        goto LABEL_21;
      v9 += 4;
    }
    while ( 1 )
    {
      v13 = *(__int64 **)(a1 + 2904);
      v14 = v11[2];
      v47 = 0;
      v46 = (unsigned __int64)v13 | 6;
      v15 = *v13;
      if ( (unsigned int)*(unsigned __int16 *)(*v13 + 68) - 1 <= 1
        && (*(_BYTE *)(*(_QWORD *)(v15 + 32) + 64LL) & 0x10) != 0 )
      {
        break;
      }
      v16 = *(_DWORD *)(v15 + 44);
      if ( (v16 & 4) != 0 || (v16 & 8) == 0 )
      {
        if ( (*(_QWORD *)(*(_QWORD *)(v15 + 16) + 24LL) & 0x100000LL) != 0 )
          break;
      }
      else if ( sub_2E88A90(v15, 0x100000, 1) )
      {
        break;
      }
      LODWORD(v12) = 0;
LABEL_7:
      HIDWORD(v47) = v12;
      sub_2F8F1B0(v14, (__int64)&v46, 1u, v10, a5, a6);
      v11 = (_QWORD *)*v11;
      if ( v11 == v9 )
        goto LABEL_19;
    }
    v17 = *(_QWORD *)v14;
    if ( (unsigned int)*(unsigned __int16 *)(*(_QWORD *)v14 + 68LL) - 1 > 1
      || (LODWORD(v12) = 1, (*(_BYTE *)(*(_QWORD *)(v17 + 32) + 64LL) & 8) == 0) )
    {
      v18 = *(_DWORD *)(v17 + 44);
      if ( (v18 & 4) != 0 || (v18 & 8) == 0 )
        v12 = (*(_QWORD *)(*(_QWORD *)(v17 + 16) + 24LL) >> 19) & 1LL;
      else
        LOBYTE(v12) = sub_2E88A90(v17, 0x80000, 1);
      LODWORD(v12) = (unsigned __int8)v12;
    }
    goto LABEL_7;
  }
LABEL_21:
  ++*(_QWORD *)a2;
  v19 = *(_DWORD *)(a2 + 8) >> 1;
  if ( v19 )
  {
    if ( (*(_BYTE *)(a2 + 8) & 1) == 0 )
    {
      v20 = 4 * v19;
      goto LABEL_24;
    }
LABEL_37:
    v22 = 8;
    v23 = (_QWORD *)(a2 + 16);
    goto LABEL_27;
  }
  if ( !*(_DWORD *)(a2 + 12) )
    goto LABEL_30;
  v20 = 0;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    goto LABEL_37;
LABEL_24:
  v21 = *(unsigned int *)(a2 + 24);
  if ( (unsigned int)v21 <= v20 || (unsigned int)v21 <= 0x40 )
  {
    v22 = 2 * v21;
    v23 = *(_QWORD **)(a2 + 16);
LABEL_27:
    for ( i = &v23[v22]; i != v23; v23 += 2 )
      *v23 = -4096;
    *(_QWORD *)(a2 + 8) &= 1uLL;
    goto LABEL_30;
  }
  if ( !v19 || (v35 = v19 - 1) == 0 )
  {
    sub_C7D6A0(*(_QWORD *)(a2 + 16), 16LL * *(unsigned int *)(a2 + 24), 8);
    *(_BYTE *)(a2 + 8) |= 1u;
    goto LABEL_40;
  }
  _BitScanReverse(&v35, v35);
  v36 = 1 << (33 - (v35 ^ 0x1F));
  if ( v36 - 5 <= 0x3A )
  {
    v36 = 64;
    sub_C7D6A0(*(_QWORD *)(a2 + 16), 16LL * *(unsigned int *)(a2 + 24), 8);
    v37 = *(_BYTE *)(a2 + 8);
    v38 = 1024;
    goto LABEL_50;
  }
  if ( (_DWORD)v21 != v36 )
  {
    sub_C7D6A0(*(_QWORD *)(a2 + 16), 16LL * *(unsigned int *)(a2 + 24), 8);
    v37 = *(_BYTE *)(a2 + 8) | 1;
    *(_BYTE *)(a2 + 8) = v37;
    if ( v36 <= 4 )
    {
LABEL_40:
      v44 = *(_QWORD *)(a2 + 8);
      *(_QWORD *)(a2 + 8) = v44 & 1;
      if ( (v44 & 1) != 0 )
      {
        v32 = (_QWORD *)(a2 + 16);
        v33 = 8;
      }
      else
      {
        v32 = *(_QWORD **)(a2 + 16);
        v33 = 2LL * *(unsigned int *)(a2 + 24);
      }
      for ( j = &v32[v33]; j != v32; v32 += 2 )
      {
        if ( v32 )
          *v32 = -4096;
      }
      goto LABEL_30;
    }
    v38 = 16LL * v36;
LABEL_50:
    *(_BYTE *)(a2 + 8) = v37 & 0xFE;
    v39 = sub_C7D670(v38, 8);
    *(_DWORD *)(a2 + 24) = v36;
    *(_QWORD *)(a2 + 16) = v39;
    goto LABEL_40;
  }
  v45 = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(a2 + 8) = v45 & 1;
  if ( (v45 & 1) != 0 )
  {
    v40 = (_QWORD *)(a2 + 16);
    v41 = 8;
  }
  else
  {
    v41 = 2 * v21;
    v40 = *(_QWORD **)(a2 + 16);
  }
  v42 = &v40[v41];
  do
  {
    if ( v40 )
      *v40 = -4096;
    v40 += 2;
  }
  while ( v42 != v40 );
LABEL_30:
  v25 = *(_QWORD ***)(a2 + 80);
  v26 = &v25[4 * *(unsigned int *)(a2 + 88)];
  v27 = v26 - 3;
  if ( v25 != v26 )
  {
    do
    {
      v28 = *v27;
      v29 = v27 - 1;
      while ( v28 != v27 )
      {
        v30 = (unsigned __int64)v28;
        v28 = (_QWORD *)*v28;
        j_j___libc_free_0(v30);
      }
      v27 -= 4;
    }
    while ( v29 != v25 );
  }
  *(_DWORD *)(a2 + 88) = 0;
  *(_DWORD *)(a2 + 224) = 0;
  return a2;
}
