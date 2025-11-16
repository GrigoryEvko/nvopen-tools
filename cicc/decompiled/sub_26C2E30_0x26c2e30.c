// Function: sub_26C2E30
// Address: 0x26c2e30
//
void __fastcall sub_26C2E30(__int64 a1)
{
  volatile signed __int32 *v2; // rdi
  _QWORD *v3; // rbx
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  _QWORD *v6; // rbx
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  _QWORD *v9; // rbx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  __int64 v13; // r14
  __int64 v14; // r13
  __int64 v15; // r14
  _QWORD *v16; // rbx
  __int64 v17; // r15
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  __int64 v20; // r14
  unsigned __int64 v21; // rdi
  __int64 v22; // r13
  __int64 v23; // r14
  _QWORD *v24; // rbx
  __int64 v25; // r15
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdi
  __int64 v28; // rdi
  __int64 v29; // r12
  unsigned __int64 v30; // rdi
  __int64 v31; // [rsp+8h] [rbp-38h]
  __int64 v32; // [rsp+8h] [rbp-38h]

  v2 = *(volatile signed __int32 **)(a1 + 336);
  if ( v2 )
    sub_A191D0(v2);
  v3 = *(_QWORD **)(a1 + 288);
  while ( v3 )
  {
    v4 = (unsigned __int64)v3;
    v3 = (_QWORD *)*v3;
    j_j___libc_free_0(v4);
  }
  memset(*(void **)(a1 + 272), 0, 8LL * *(_QWORD *)(a1 + 280));
  v5 = *(_QWORD *)(a1 + 272);
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  if ( v5 != a1 + 320 )
    j_j___libc_free_0(v5);
  v6 = *(_QWORD **)(a1 + 216);
  while ( v6 )
  {
    v7 = (unsigned __int64)v6;
    v6 = (_QWORD *)*v6;
    j_j___libc_free_0(v7);
  }
  memset(*(void **)(a1 + 200), 0, 8LL * *(_QWORD *)(a1 + 208));
  v8 = *(_QWORD *)(a1 + 200);
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  if ( v8 != a1 + 248 )
    j_j___libc_free_0(v8);
  v9 = *(_QWORD **)(a1 + 160);
  while ( v9 )
  {
    v10 = (unsigned __int64)v9;
    v9 = (_QWORD *)*v9;
    j_j___libc_free_0(v10);
  }
  memset(*(void **)(a1 + 144), 0, 8LL * *(_QWORD *)(a1 + 152));
  v11 = *(_QWORD *)(a1 + 144);
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  if ( v11 != a1 + 192 )
    j_j___libc_free_0(v11);
  v12 = *(_QWORD *)(a1 + 120);
  if ( *(_DWORD *)(a1 + 132) )
  {
    v13 = *(unsigned int *)(a1 + 128);
    if ( (_DWORD)v13 )
    {
      v14 = 0;
      v31 = 8 * v13;
      do
      {
        v15 = *(_QWORD *)(v12 + v14);
        if ( v15 != -8 && v15 )
        {
          v16 = *(_QWORD **)(v15 + 24);
          v17 = *(_QWORD *)v15 + 65LL;
          while ( v16 )
          {
            v18 = (unsigned __int64)v16;
            v16 = (_QWORD *)*v16;
            j_j___libc_free_0(v18);
          }
          memset(*(void **)(v15 + 8), 0, 8LL * *(_QWORD *)(v15 + 16));
          v19 = *(_QWORD *)(v15 + 8);
          *(_QWORD *)(v15 + 32) = 0;
          *(_QWORD *)(v15 + 24) = 0;
          if ( v19 != v15 + 56 )
            j_j___libc_free_0(v19);
          sub_C7D6A0(v15, v17, 8);
          v12 = *(_QWORD *)(a1 + 120);
        }
        v14 += 8;
      }
      while ( v31 != v14 );
    }
  }
  _libc_free(v12);
  if ( *(_DWORD *)(a1 + 108) )
  {
    v20 = *(unsigned int *)(a1 + 104);
    v21 = *(_QWORD *)(a1 + 96);
    if ( (_DWORD)v20 )
    {
      v22 = 0;
      v32 = 8 * v20;
      do
      {
        v23 = *(_QWORD *)(v21 + v22);
        if ( v23 != -8 && v23 )
        {
          v24 = *(_QWORD **)(v23 + 24);
          v25 = *(_QWORD *)v23 + 65LL;
          while ( v24 )
          {
            v26 = (unsigned __int64)v24;
            v24 = (_QWORD *)*v24;
            j_j___libc_free_0(v26);
          }
          memset(*(void **)(v23 + 8), 0, 8LL * *(_QWORD *)(v23 + 16));
          v27 = *(_QWORD *)(v23 + 8);
          *(_QWORD *)(v23 + 32) = 0;
          *(_QWORD *)(v23 + 24) = 0;
          if ( v27 != v23 + 56 )
            j_j___libc_free_0(v27);
          sub_C7D6A0(v23, v25, 8);
          v21 = *(_QWORD *)(a1 + 96);
        }
        v22 += 8;
      }
      while ( v22 != v32 );
    }
  }
  else
  {
    v21 = *(_QWORD *)(a1 + 96);
  }
  _libc_free(v21);
  v28 = a1 + 40;
  v29 = a1 + 88;
  sub_26C2AF0(v28);
  v30 = *(_QWORD *)(v29 - 48);
  if ( v30 != v29 )
    j_j___libc_free_0(v30);
}
