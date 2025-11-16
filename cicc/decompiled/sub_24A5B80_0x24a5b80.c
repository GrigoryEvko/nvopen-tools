// Function: sub_24A5B80
// Address: 0x24a5b80
//
__int64 __fastcall sub_24A5B80(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // r15
  unsigned __int64 *v4; // r14
  unsigned __int64 v5; // rdi
  unsigned __int64 *v6; // r12
  unsigned __int64 *v7; // rbx
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  __int64 v10; // rsi
  _QWORD *v11; // rbx
  _QWORD *v12; // r12
  unsigned __int64 v13; // r14
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  __int64 v16; // rsi
  unsigned __int64 *v17; // rbx
  unsigned __int64 *v18; // r12
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 *v21; // rbx
  unsigned __int64 *v22; // r12
  unsigned __int64 v23; // rdi
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 v27; // r12
  unsigned __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rbx
  __int64 v31; // r12
  unsigned __int64 v32; // rdi

  sub_24DABD0(a1 + 512, a2);
  v3 = *(_QWORD *)(a1 + 496);
  if ( v3 )
  {
    v4 = (unsigned __int64 *)(v3 + 72);
    do
    {
      v5 = *(v4 - 3);
      v6 = (unsigned __int64 *)*(v4 - 2);
      v4 -= 3;
      v7 = (unsigned __int64 *)v5;
      if ( v6 != (unsigned __int64 *)v5 )
      {
        do
        {
          if ( *v7 )
            j_j___libc_free_0(*v7);
          v7 += 3;
        }
        while ( v6 != v7 );
        v5 = *v4;
      }
      if ( v5 )
        j_j___libc_free_0(v5);
    }
    while ( (unsigned __int64 *)v3 != v4 );
    j_j___libc_free_0(v3);
  }
  v8 = *(_QWORD *)(a1 + 472);
  if ( v8 )
    j_j___libc_free_0(v8);
  v9 = *(_QWORD *)(a1 + 448);
  if ( v9 )
    j_j___libc_free_0(v9);
  if ( *(_BYTE *)(a1 + 424) )
  {
    v25 = *(unsigned int *)(a1 + 416);
    *(_BYTE *)(a1 + 424) = 0;
    if ( (_DWORD)v25 )
    {
      v26 = *(_QWORD *)(a1 + 400);
      v27 = v26 + 88 * v25;
      do
      {
        if ( *(_QWORD *)v26 != -8192 && *(_QWORD *)v26 != -4096 )
        {
          v28 = *(_QWORD *)(v26 + 40);
          if ( v28 != v26 + 56 )
            _libc_free(v28);
          sub_C7D6A0(*(_QWORD *)(v26 + 16), 8LL * *(unsigned int *)(v26 + 32), 8);
        }
        v26 += 88;
      }
      while ( v27 != v26 );
      v25 = *(unsigned int *)(a1 + 416);
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 400), 88 * v25, 8);
    v29 = *(unsigned int *)(a1 + 384);
    if ( (_DWORD)v29 )
    {
      v30 = *(_QWORD *)(a1 + 368);
      v31 = v30 + 88 * v29;
      do
      {
        if ( *(_QWORD *)v30 != -8192 && *(_QWORD *)v30 != -4096 )
        {
          v32 = *(_QWORD *)(v30 + 40);
          if ( v32 != v30 + 56 )
            _libc_free(v32);
          sub_C7D6A0(*(_QWORD *)(v30 + 16), 8LL * *(unsigned int *)(v30 + 32), 8);
        }
        v30 += 88;
      }
      while ( v31 != v30 );
      v29 = *(unsigned int *)(a1 + 384);
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 368), 88 * v29, 8);
  }
  v10 = *(unsigned int *)(a1 + 296);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD **)(a1 + 280);
    v12 = &v11[2 * v10];
    do
    {
      if ( *v11 != -4096 && *v11 != -8192 )
      {
        v13 = v11[1];
        if ( v13 )
        {
          v14 = *(_QWORD *)(v13 + 72);
          if ( v14 != v13 + 88 )
            _libc_free(v14);
          v15 = *(_QWORD *)(v13 + 40);
          if ( v15 != v13 + 56 )
            _libc_free(v15);
          j_j___libc_free_0(v13);
        }
      }
      v11 += 2;
    }
    while ( v12 != v11 );
    v10 = *(unsigned int *)(a1 + 296);
  }
  v16 = 16 * v10;
  sub_C7D6A0(*(_QWORD *)(a1 + 280), v16, 8);
  v17 = *(unsigned __int64 **)(a1 + 256);
  v18 = *(unsigned __int64 **)(a1 + 248);
  if ( v17 != v18 )
  {
    do
    {
      if ( *v18 )
      {
        v16 = 48;
        j_j___libc_free_0(*v18);
      }
      ++v18;
    }
    while ( v17 != v18 );
    v18 = *(unsigned __int64 **)(a1 + 248);
  }
  if ( v18 )
  {
    v16 = *(_QWORD *)(a1 + 264) - (_QWORD)v18;
    j_j___libc_free_0((unsigned __int64)v18);
  }
  v19 = *(_QWORD *)(a1 + 192);
  if ( v19 != a1 + 208 )
  {
    v16 = *(_QWORD *)(a1 + 208) + 1LL;
    j_j___libc_free_0(v19);
  }
  v20 = *(_QWORD *)(a1 + 160);
  if ( v20 != a1 + 176 )
  {
    v16 = *(_QWORD *)(a1 + 176) + 1LL;
    j_j___libc_free_0(v20);
  }
  v21 = *(unsigned __int64 **)(a1 + 80);
  v22 = *(unsigned __int64 **)(a1 + 72);
  if ( v21 != v22 )
  {
    do
    {
      v23 = *v22;
      if ( *v22 )
      {
        v16 = v22[2] - v23;
        j_j___libc_free_0(v23);
      }
      v22 += 3;
    }
    while ( v21 != v22 );
    v22 = *(unsigned __int64 **)(a1 + 72);
  }
  if ( v22 )
  {
    v16 = *(_QWORD *)(a1 + 88) - (_QWORD)v22;
    j_j___libc_free_0((unsigned __int64)v22);
  }
  return sub_24DABD0(a1 + 56, v16);
}
