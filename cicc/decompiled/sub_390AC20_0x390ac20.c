// Function: sub_390AC20
// Address: 0x390ac20
//
void __fastcall sub_390AC20(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int64 **v6; // r14
  unsigned __int64 **v7; // r13
  unsigned __int64 *v8; // rbx
  unsigned __int64 *v9; // r15
  unsigned __int64 *v10; // r13
  unsigned __int64 *v11; // r14
  unsigned __int64 *v12; // rbx
  void *v13; // rdi
  unsigned int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rbx
  unsigned __int64 v19; // rdi
  __int64 v20; // rdi
  void (*v21)(void); // rax
  __int64 v22; // rdi
  void (*v23)(void); // rax
  __int64 v24; // rdi
  void (*v25)(void); // rax
  __int64 v26; // r13
  __int64 v27; // rbx
  unsigned __int64 v28; // rdi
  unsigned __int64 **v29; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 32);
  if ( v2 != *(_QWORD *)(a1 + 40) )
    *(_QWORD *)(a1 + 40) = v2;
  v3 = *(_QWORD *)(a1 + 56);
  if ( v3 != *(_QWORD *)(a1 + 64) )
    *(_QWORD *)(a1 + 64) = v3;
  v4 = *(_QWORD *)(a1 + 80);
  if ( v4 != *(_QWORD *)(a1 + 88) )
    *(_QWORD *)(a1 + 88) = v4;
  v5 = *(_QWORD *)(a1 + 104);
  if ( *(_QWORD *)(a1 + 112) != v5 )
    *(_QWORD *)(a1 + 112) = v5;
  v6 = *(unsigned __int64 ***)(a1 + 136);
  v29 = *(unsigned __int64 ***)(a1 + 128);
  if ( v29 != v6 )
  {
    v7 = *(unsigned __int64 ***)(a1 + 128);
    do
    {
      v8 = v7[1];
      v9 = *v7;
      if ( v8 != *v7 )
      {
        do
        {
          if ( (unsigned __int64 *)*v9 != v9 + 2 )
            j_j___libc_free_0(*v9);
          v9 += 4;
        }
        while ( v8 != v9 );
        v9 = *v7;
      }
      if ( v9 )
        j_j___libc_free_0((unsigned __int64)v9);
      v7 += 3;
    }
    while ( v6 != v7 );
    *(_QWORD *)(a1 + 136) = v29;
  }
  v10 = *(unsigned __int64 **)(a1 + 152);
  v11 = *(unsigned __int64 **)(a1 + 160);
  if ( v10 != v11 )
  {
    v12 = *(unsigned __int64 **)(a1 + 152);
    do
    {
      if ( (unsigned __int64 *)*v12 != v12 + 2 )
        j_j___libc_free_0(*v12);
      v12 += 4;
    }
    while ( v11 != v12 );
    *(_QWORD *)(a1 + 160) = v10;
  }
  ++*(_QWORD *)(a1 + 184);
  v13 = *(void **)(a1 + 200);
  if ( v13 != *(void **)(a1 + 192) )
  {
    v14 = 4 * (*(_DWORD *)(a1 + 212) - *(_DWORD *)(a1 + 216));
    v15 = *(unsigned int *)(a1 + 208);
    if ( v14 < 0x20 )
      v14 = 32;
    if ( (unsigned int)v15 > v14 )
    {
      sub_16CC920(a1 + 184);
      goto LABEL_32;
    }
    memset(v13, -1, 8 * v15);
  }
  *(_QWORD *)(a1 + 212) = 0;
LABEL_32:
  v16 = *(unsigned int *)(a1 + 512);
  *(_BYTE *)(a1 + 484) &= 0xF8u;
  v17 = *(_QWORD *)(a1 + 504);
  *(_DWORD *)(a1 + 480) = 0;
  *(_DWORD *)(a1 + 488) = 0;
  v18 = v17 + 48 * v16;
  while ( v17 != v18 )
  {
    while ( 1 )
    {
      v18 -= 48;
      v19 = *(_QWORD *)(v18 + 8);
      if ( v19 == v18 + 24 )
        break;
      _libc_free(v19);
      if ( v17 == v18 )
        goto LABEL_36;
    }
  }
LABEL_36:
  v20 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 512) = 0;
  *(_QWORD *)(a1 + 496) = 0;
  *(_DWORD *)(a1 + 2064) = 0;
  if ( v20 )
  {
    v21 = *(void (**)(void))(*(_QWORD *)v20 + 16LL);
    if ( v21 != nullsub_1965 )
      v21();
  }
  v22 = *(_QWORD *)(a1 + 16);
  if ( v22 )
  {
    v23 = *(void (**)(void))(*(_QWORD *)v22 + 16LL);
    if ( v23 != nullsub_1968 )
      v23();
  }
  v24 = *(_QWORD *)(a1 + 24);
  if ( v24 )
  {
    v25 = *(void (**)(void))(*(_QWORD *)v24 + 16LL);
    if ( v25 != nullsub_1934 )
      v25();
  }
  v26 = *(_QWORD *)(a1 + 504);
  v27 = v26 + 48LL * *(unsigned int *)(a1 + 512);
  while ( v26 != v27 )
  {
    while ( 1 )
    {
      v27 -= 48;
      v28 = *(_QWORD *)(v27 + 8);
      if ( v28 == v27 + 24 )
        break;
      _libc_free(v28);
      if ( v26 == v27 )
        goto LABEL_49;
    }
  }
LABEL_49:
  *(_DWORD *)(a1 + 512) = 0;
  *(_QWORD *)(a1 + 496) = 0;
}
