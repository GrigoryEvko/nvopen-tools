// Function: sub_2A1CC40
// Address: 0x2a1cc40
//
void __fastcall sub_2A1CC40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // r13
  _QWORD *v6; // rax
  unsigned __int64 v7; // r9
  _QWORD *v8; // rbx
  int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // r8
  unsigned __int64 v12; // rcx
  __int64 v13; // rdx
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned int v18; // edx
  unsigned int v19; // r15d
  __int64 v20; // r14
  _QWORD *v21; // rax
  _QWORD *v22; // r14
  _QWORD *v23; // rax
  __int64 v24; // rax
  __int64 v25; // r14
  unsigned __int64 v26; // rdi
  __int64 v27; // [rsp+8h] [rbp-48h]
  unsigned __int64 v28; // [rsp+10h] [rbp-40h]
  __int64 v29; // [rsp+10h] [rbp-40h]
  __int64 v30; // [rsp+18h] [rbp-38h]
  __int64 v31; // [rsp+18h] [rbp-38h]
  __int64 v32; // [rsp+18h] [rbp-38h]

  if ( a3 )
  {
    v4 = (unsigned int)(*(_DWORD *)(a3 + 44) + 1);
    if ( (unsigned int)v4 < *(_DWORD *)(a1 + 32) )
      goto LABEL_3;
LABEL_33:
    *(_BYTE *)(a1 + 112) = 0;
    v23 = (_QWORD *)sub_22077B0(0x50u);
    v8 = v23;
    if ( !v23 )
    {
      v5 = 0;
      goto LABEL_7;
    }
    *v23 = a2;
    v5 = 0;
    v9 = 0;
    v8[1] = 0;
    goto LABEL_6;
  }
  v4 = 0;
  if ( !*(_DWORD *)(a1 + 32) )
    goto LABEL_33;
LABEL_3:
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v4);
  *(_BYTE *)(a1 + 112) = 0;
  v6 = (_QWORD *)sub_22077B0(0x50u);
  v8 = v6;
  if ( v6 )
  {
    *v6 = a2;
    v6[1] = v5;
    if ( v5 )
      v9 = *(_DWORD *)(v5 + 16) + 1;
    else
      v9 = 0;
LABEL_6:
    *((_DWORD *)v8 + 4) = v9;
    v8[3] = v8 + 5;
    v8[4] = 0x400000000LL;
    v8[9] = -1;
  }
LABEL_7:
  if ( a2 )
  {
    v10 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
    v11 = 8 * v10;
  }
  else
  {
    v11 = 0;
    LODWORD(v10) = 0;
  }
  v12 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)v12 > (unsigned int)v10 )
    goto LABEL_10;
  v17 = *(_QWORD *)(a1 + 104);
  v18 = v10 + 1;
  if ( *(_DWORD *)(v17 + 88) >= v18 )
    v18 = *(_DWORD *)(v17 + 88);
  v19 = v18;
  if ( v18 == v12 )
  {
LABEL_10:
    v13 = *(_QWORD *)(a1 + 24);
    goto LABEL_11;
  }
  v20 = 8LL * v18;
  if ( v18 < v12 )
  {
    v13 = *(_QWORD *)(a1 + 24);
    v24 = v13 + 8 * v12;
    v25 = v13 + v20;
    if ( v24 == v25 )
      goto LABEL_31;
    do
    {
      v7 = *(_QWORD *)(v24 - 8);
      v24 -= 8;
      if ( v7 )
      {
        v26 = *(_QWORD *)(v7 + 24);
        if ( v26 != v7 + 40 )
        {
          v27 = v11;
          v28 = v7;
          v31 = v24;
          _libc_free(v26);
          v11 = v27;
          v7 = v28;
          v24 = v31;
        }
        v29 = v11;
        v32 = v24;
        j_j___libc_free_0(v7);
        v11 = v29;
        v24 = v32;
      }
    }
    while ( v25 != v24 );
  }
  else
  {
    if ( v18 > (unsigned __int64)*(unsigned int *)(a1 + 36) )
    {
      v30 = v11;
      sub_B1B4E0(a1 + 24, v18);
      v12 = *(unsigned int *)(a1 + 32);
      v11 = v30;
    }
    v13 = *(_QWORD *)(a1 + 24);
    v21 = (_QWORD *)(v13 + 8 * v12);
    v22 = (_QWORD *)(v13 + v20);
    if ( v21 == v22 )
      goto LABEL_31;
    do
    {
      if ( v21 )
        *v21 = 0;
      ++v21;
    }
    while ( v22 != v21 );
  }
  v13 = *(_QWORD *)(a1 + 24);
LABEL_31:
  *(_DWORD *)(a1 + 32) = v19;
LABEL_11:
  v14 = *(_QWORD *)(v13 + v11);
  *(_QWORD *)(v13 + v11) = v8;
  if ( v14 )
  {
    v15 = *(_QWORD *)(v14 + 24);
    if ( v15 != v14 + 40 )
      _libc_free(v15);
    j_j___libc_free_0(v14);
  }
  if ( v5 )
  {
    v16 = *(unsigned int *)(v5 + 32);
    if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(v5 + 36) )
    {
      sub_C8D5F0(v5 + 24, (const void *)(v5 + 40), v16 + 1, 8u, v11, v7);
      v16 = *(unsigned int *)(v5 + 32);
    }
    *(_QWORD *)(*(_QWORD *)(v5 + 24) + 8 * v16) = v8;
    ++*(_DWORD *)(v5 + 32);
  }
}
