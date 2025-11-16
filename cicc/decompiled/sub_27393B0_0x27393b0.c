// Function: sub_27393B0
// Address: 0x27393b0
//
void __fastcall sub_27393B0(__int64 a1, int *a2, unsigned int *a3, unsigned __int8 *a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  unsigned int v10; // r15d
  __int64 v11; // rdx
  int v12; // r14d
  __int64 v13; // rcx
  __int64 v14; // rdi
  char *v15; // rdi
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // r14
  __int64 v20; // rcx
  int v21; // eax
  int v22; // r10d
  char **v23; // r8
  __int64 v24; // rdx
  __int64 v25; // rdi
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // r12
  __int64 v28; // r13
  char v29; // dl
  unsigned __int64 v30; // r13
  unsigned __int64 v31; // rdi
  int v32; // r13d
  unsigned __int8 v33; // [rsp+Bh] [rbp-75h]
  unsigned int v34; // [rsp+Ch] [rbp-74h]
  unsigned __int8 v35; // [rsp+10h] [rbp-70h]
  int v37; // [rsp+10h] [rbp-70h]
  unsigned __int64 v38; // [rsp+18h] [rbp-68h]
  unsigned __int64 v39; // [rsp+28h] [rbp-58h] BYREF
  char *v40; // [rsp+30h] [rbp-50h] BYREF
  __int64 v41; // [rsp+38h] [rbp-48h]
  _BYTE v42[64]; // [rsp+40h] [rbp-40h] BYREF

  v9 = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)v9 < *(_DWORD *)(a1 + 12) )
  {
    v10 = *a3;
    v11 = *a4;
    v40 = v42;
    v12 = *a2;
    v41 = 0x200000000LL;
    v13 = *(unsigned int *)(a5 + 8);
    if ( (_DWORD)v13 )
    {
      v35 = v11;
      sub_2738630((__int64)&v40, (char **)a5, v11, v13, (__int64)&v40, a6);
      v9 = *(unsigned int *)(a1 + 8);
      v11 = v35;
    }
    v14 = *(_QWORD *)a1 + 48 * v9;
    if ( v14
      && (*(_DWORD *)v14 = v12,
          *(_QWORD *)(v14 + 16) = v14 + 32,
          *(_DWORD *)(v14 + 4) = v10,
          *(_BYTE *)(v14 + 8) = v11,
          *(_QWORD *)(v14 + 24) = 0x200000000LL,
          (_DWORD)v41) )
    {
      sub_2738630(v14 + 16, &v40, v11, v13, (__int64)&v40, a6);
      v15 = v40;
      if ( v40 == v42 )
        goto LABEL_8;
    }
    else
    {
      v15 = v40;
      if ( v40 == v42 )
      {
LABEL_8:
        ++*(_DWORD *)(a1 + 8);
        return;
      }
    }
    _libc_free((unsigned __int64)v15);
    goto LABEL_8;
  }
  v16 = a1 + 16;
  v17 = sub_C8D7D0(a1, a1 + 16, 0, 0x30u, &v39, a6);
  v18 = (__int64)a3;
  v19 = v17;
  v20 = *a4;
  v41 = 0x200000000LL;
  v21 = *a2;
  v22 = *(_DWORD *)(a5 + 8);
  v40 = v42;
  v23 = &v40;
  v24 = *a3;
  if ( v22 )
  {
    v33 = v20;
    v34 = *a3;
    v37 = v21;
    sub_2738630((__int64)&v40, (char **)a5, v24, v20, (__int64)&v40, v18);
    v20 = v33;
    v24 = v34;
    v21 = v37;
    v23 = &v40;
  }
  v25 = v19 + 48LL * *(unsigned int *)(a1 + 8);
  if ( v25 )
  {
    *(_DWORD *)v25 = v21;
    *(_QWORD *)(v25 + 16) = v25 + 32;
    *(_DWORD *)(v25 + 4) = v24;
    *(_BYTE *)(v25 + 8) = v20;
    *(_QWORD *)(v25 + 24) = 0x200000000LL;
    v18 = (unsigned int)v41;
    if ( (_DWORD)v41 )
      sub_2738630(v25 + 16, &v40, v24, v20, (__int64)&v40, (unsigned int)v41);
  }
  if ( v40 != v42 )
    _libc_free((unsigned __int64)v40);
  v26 = *(_QWORD *)a1;
  v27 = *(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v27 )
  {
    v28 = v19;
    do
    {
      if ( v28 )
      {
        *(_DWORD *)v28 = *(_DWORD *)v26;
        *(_DWORD *)(v28 + 4) = *(_DWORD *)(v26 + 4);
        v29 = *(_BYTE *)(v26 + 8);
        *(_DWORD *)(v28 + 24) = 0;
        *(_BYTE *)(v28 + 8) = v29;
        *(_QWORD *)(v28 + 16) = v28 + 32;
        *(_DWORD *)(v28 + 28) = 2;
        if ( *(_DWORD *)(v26 + 24) )
        {
          v38 = v26;
          sub_2738630(v28 + 16, (char **)(v26 + 16), v28 + 32, v20, (__int64)v23, v18);
          v26 = v38;
        }
      }
      v26 += 48LL;
      v28 += 48;
    }
    while ( v27 != v26 );
    v30 = *(_QWORD *)a1;
    v27 = *(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v27 )
    {
      do
      {
        v27 -= 48LL;
        v31 = *(_QWORD *)(v27 + 16);
        if ( v31 != v27 + 32 )
          _libc_free(v31);
      }
      while ( v30 != v27 );
      v27 = *(_QWORD *)a1;
    }
  }
  v32 = v39;
  if ( v16 != v27 )
    _libc_free(v27);
  *(_QWORD *)a1 = v19;
  *(_DWORD *)(a1 + 12) = v32;
  ++*(_DWORD *)(a1 + 8);
}
