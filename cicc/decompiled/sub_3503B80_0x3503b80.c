// Function: sub_3503B80
// Address: 0x3503b80
//
void __fastcall sub_3503B80(unsigned __int64 a1, __int64 a2)
{
  __int64 *v3; // r12
  __int64 v4; // r15
  __int64 *v5; // rbx
  __int64 *v6; // r14
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // rdx
  char v12; // al
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  __int64 v15; // rax
  __int64 *v16; // rbx
  __int64 *v17; // r12
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rdx
  unsigned __int64 v21; // r12
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  __int64 *v24; // rax
  __int64 v25; // rcx
  __int64 *v26; // rbx
  __int64 *v27; // r14
  __int64 v28; // rdi
  unsigned int v29; // ecx
  __int64 v30; // rsi
  __int64 *v31; // rbx
  __int64 v32; // rsi
  __int64 v33; // rdi
  __int64 *v34; // [rsp+8h] [rbp-38h]

  sub_301D560(a1);
  v3 = *(__int64 **)(a1 + 32);
  v34 = *(__int64 **)(a1 + 40);
  if ( v3 != v34 )
  {
    do
    {
      v4 = *v3;
      v5 = *(__int64 **)(*v3 + 16);
      if ( *(__int64 **)(*v3 + 8) == v5 )
      {
        *(_BYTE *)(v4 + 152) = 1;
      }
      else
      {
        v6 = *(__int64 **)(*v3 + 8);
        do
        {
          v7 = *v6++;
          sub_2EA4EF0(v7, a2);
        }
        while ( v5 != v6 );
        *(_BYTE *)(v4 + 152) = 1;
        v8 = *(_QWORD *)(v4 + 8);
        if ( v8 != *(_QWORD *)(v4 + 16) )
          *(_QWORD *)(v4 + 16) = v8;
      }
      v9 = *(_QWORD *)(v4 + 32);
      if ( v9 != *(_QWORD *)(v4 + 40) )
        *(_QWORD *)(v4 + 40) = v9;
      ++*(_QWORD *)(v4 + 56);
      if ( *(_BYTE *)(v4 + 84) )
      {
        *(_QWORD *)v4 = 0;
      }
      else
      {
        v10 = 4 * (*(_DWORD *)(v4 + 76) - *(_DWORD *)(v4 + 80));
        v11 = *(unsigned int *)(v4 + 72);
        if ( v10 < 0x20 )
          v10 = 32;
        if ( (unsigned int)v11 > v10 )
        {
          sub_C8C990(v4 + 56, a2);
        }
        else
        {
          a2 = 0xFFFFFFFFLL;
          memset(*(void **)(v4 + 64), -1, 8 * v11);
        }
        v12 = *(_BYTE *)(v4 + 84);
        *(_QWORD *)v4 = 0;
        if ( !v12 )
          _libc_free(*(_QWORD *)(v4 + 64));
      }
      v13 = *(_QWORD *)(v4 + 32);
      if ( v13 )
      {
        a2 = *(_QWORD *)(v4 + 48) - v13;
        j_j___libc_free_0(v13);
      }
      v14 = *(_QWORD *)(v4 + 8);
      if ( v14 )
      {
        a2 = *(_QWORD *)(v4 + 24) - v14;
        j_j___libc_free_0(v14);
      }
      ++v3;
    }
    while ( v34 != v3 );
    v15 = *(_QWORD *)(a1 + 32);
    if ( *(_QWORD *)(a1 + 40) != v15 )
      *(_QWORD *)(a1 + 40) = v15;
  }
  v16 = *(__int64 **)(a1 + 120);
  v17 = &v16[2 * *(unsigned int *)(a1 + 128)];
  while ( v17 != v16 )
  {
    v18 = v16[1];
    v19 = *v16;
    v16 += 2;
    sub_C7D6A0(v19, v18, 16);
  }
  *(_DWORD *)(a1 + 128) = 0;
  v20 = *(unsigned int *)(a1 + 80);
  if ( !(_DWORD)v20 )
    goto LABEL_26;
  *(_QWORD *)(a1 + 136) = 0;
  v24 = *(__int64 **)(a1 + 72);
  v25 = *v24;
  v26 = &v24[v20];
  v27 = v24 + 1;
  *(_QWORD *)(a1 + 56) = *v24;
  for ( *(_QWORD *)(a1 + 64) = v25 + 4096; v26 != v27; v24 = *(__int64 **)(a1 + 72) )
  {
    v28 = *v27;
    v29 = (unsigned int)(v27 - v24) >> 7;
    v30 = 4096LL << v29;
    if ( v29 >= 0x1E )
      v30 = 0x40000000000LL;
    ++v27;
    sub_C7D6A0(v28, v30, 16);
  }
  *(_DWORD *)(a1 + 80) = 1;
  sub_C7D6A0(*v24, 4096, 16);
  v31 = *(__int64 **)(a1 + 120);
  v21 = (unsigned __int64)&v31[2 * *(unsigned int *)(a1 + 128)];
  if ( v31 != (__int64 *)v21 )
  {
    do
    {
      v32 = v31[1];
      v33 = *v31;
      v31 += 2;
      sub_C7D6A0(v33, v32, 16);
    }
    while ( (__int64 *)v21 != v31 );
LABEL_26:
    v21 = *(_QWORD *)(a1 + 120);
  }
  if ( v21 != a1 + 136 )
    _libc_free(v21);
  v22 = *(_QWORD *)(a1 + 72);
  if ( v22 != a1 + 88 )
    _libc_free(v22);
  v23 = *(_QWORD *)(a1 + 32);
  if ( v23 )
    j_j___libc_free_0(v23);
  sub_C7D6A0(*(_QWORD *)(a1 + 8), 16LL * *(unsigned int *)(a1 + 24), 8);
  j_j___libc_free_0(a1);
}
