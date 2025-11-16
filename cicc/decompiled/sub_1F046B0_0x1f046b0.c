// Function: sub_1F046B0
// Address: 0x1f046b0
//
__int64 __fastcall sub_1F046B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, unsigned __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rcx
  _QWORD *v9; // rbx
  _QWORD *v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int16 v17; // dx
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int16 v20; // ax
  int v21; // eax
  __int64 v22; // rdx
  _QWORD *v23; // rax
  _QWORD *i; // rdx
  _QWORD **v25; // r15
  _QWORD **v26; // r13
  _QWORD **k; // r12
  _QWORD *v28; // rbx
  _QWORD *v29; // rdi
  unsigned int v31; // ecx
  _QWORD *v32; // rdi
  unsigned int v33; // eax
  int v34; // eax
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rax
  int v37; // ebx
  __int64 v38; // r12
  _QWORD *v39; // rax
  __int64 v40; // rdx
  _QWORD *j; // rdx
  _QWORD *v42; // rax
  _QWORD *v43; // [rsp+8h] [rbp-48h]
  __int64 v44; // [rsp+10h] [rbp-40h] BYREF
  __int64 v45; // [rsp+18h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 40);
  v43 = (_QWORD *)v8;
  v9 = (_QWORD *)(v7 + 8);
  if ( v7 != v8 )
  {
    while ( 1 )
    {
      v10 = (_QWORD *)*v9;
      if ( (_QWORD *)*v9 != v9 )
        break;
LABEL_19:
      if ( v43 == v9 + 3 )
        goto LABEL_21;
      v9 += 4;
    }
    while ( 1 )
    {
      v12 = *(_QWORD *)(a1 + 1984);
      v13 = v10[2];
      v45 = 0;
      v14 = *(_QWORD *)(v12 + 8);
      v15 = v12 | 6;
      v16 = *(_QWORD *)(v14 + 16);
      v44 = v15;
      if ( *(_WORD *)v16 == 1 && (*(_BYTE *)(*(_QWORD *)(v14 + 32) + 64LL) & 0x10) != 0 )
        break;
      v17 = *(_WORD *)(v14 + 46);
      if ( (v17 & 4) != 0 || (v17 & 8) == 0 )
      {
        if ( (*(_QWORD *)(v16 + 8) & 0x20000LL) != 0 )
          break;
      }
      else if ( sub_1E15D00(v14, 0x20000u, 1) )
      {
        break;
      }
      LODWORD(v11) = 0;
LABEL_7:
      HIDWORD(v45) = v11;
      sub_1F01A00(v13, (__int64)&v44, 1, v8, a5, a6);
      v10 = (_QWORD *)*v10;
      if ( v10 == v9 )
        goto LABEL_19;
    }
    v18 = *(_QWORD *)(v13 + 8);
    v19 = *(_QWORD *)(v18 + 16);
    if ( *(_WORD *)v19 != 1 || (LODWORD(v11) = 1, (*(_BYTE *)(*(_QWORD *)(v18 + 32) + 64LL) & 8) == 0) )
    {
      v20 = *(_WORD *)(v18 + 46);
      if ( (v20 & 4) != 0 || (v20 & 8) == 0 )
        v11 = (*(_QWORD *)(v19 + 8) >> 16) & 1LL;
      else
        LOBYTE(v11) = sub_1E15D00(v18, 0x10000u, 1);
      LODWORD(v11) = (unsigned __int8)v11;
    }
    goto LABEL_7;
  }
LABEL_21:
  ++*(_QWORD *)a2;
  v21 = *(_DWORD *)(a2 + 16);
  if ( !v21 )
  {
    if ( !*(_DWORD *)(a2 + 20) )
      goto LABEL_27;
    v22 = *(unsigned int *)(a2 + 24);
    if ( (unsigned int)v22 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a2 + 8));
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)(a2 + 16) = 0;
      *(_DWORD *)(a2 + 24) = 0;
      goto LABEL_27;
    }
    goto LABEL_24;
  }
  v22 = *(unsigned int *)(a2 + 24);
  v31 = 4 * v21;
  if ( (unsigned int)(4 * v21) < 0x40 )
    v31 = 64;
  if ( v31 >= (unsigned int)v22 )
  {
LABEL_24:
    v23 = *(_QWORD **)(a2 + 8);
    for ( i = &v23[2 * v22]; i != v23; v23 += 2 )
      *v23 = -8;
    *(_QWORD *)(a2 + 16) = 0;
    goto LABEL_27;
  }
  v32 = *(_QWORD **)(a2 + 8);
  v33 = v21 - 1;
  if ( !v33 )
  {
    v38 = 2048;
    v37 = 128;
LABEL_43:
    j___libc_free_0(v32);
    *(_DWORD *)(a2 + 24) = v37;
    v39 = (_QWORD *)sub_22077B0(v38);
    v40 = *(unsigned int *)(a2 + 24);
    *(_QWORD *)(a2 + 16) = 0;
    *(_QWORD *)(a2 + 8) = v39;
    for ( j = &v39[2 * v40]; j != v39; v39 += 2 )
    {
      if ( v39 )
        *v39 = -8;
    }
    goto LABEL_27;
  }
  _BitScanReverse(&v33, v33);
  v34 = 1 << (33 - (v33 ^ 0x1F));
  if ( v34 < 64 )
    v34 = 64;
  if ( (_DWORD)v22 != v34 )
  {
    v35 = (4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1);
    v36 = ((v35 | (v35 >> 2)) >> 4) | v35 | (v35 >> 2) | ((((v35 | (v35 >> 2)) >> 4) | v35 | (v35 >> 2)) >> 8);
    v37 = (v36 | (v36 >> 16)) + 1;
    v38 = 16 * ((v36 | (v36 >> 16)) + 1);
    goto LABEL_43;
  }
  *(_QWORD *)(a2 + 16) = 0;
  v42 = &v32[2 * (unsigned int)v22];
  do
  {
    if ( v32 )
      *v32 = -8;
    v32 += 2;
  }
  while ( v42 != v32 );
LABEL_27:
  v25 = *(_QWORD ***)(a2 + 32);
  v26 = *(_QWORD ***)(a2 + 40);
  if ( v25 != v26 )
  {
    for ( k = v25 + 1; ; k += 4 )
    {
      v28 = *k;
      while ( v28 != k )
      {
        v29 = v28;
        v28 = (_QWORD *)*v28;
        j_j___libc_free_0(v29, 24);
      }
      if ( v26 == k + 3 )
        break;
    }
    *(_QWORD *)(a2 + 40) = v25;
  }
  *(_DWORD *)(a2 + 56) = 0;
  return a2;
}
