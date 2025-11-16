// Function: sub_35041B0
// Address: 0x35041b0
//
void __fastcall sub_35041B0(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // r12
  unsigned __int64 v3; // r13
  __int64 *v4; // r12
  __int64 v5; // r15
  __int64 *v6; // rbx
  __int64 *v7; // r14
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned int v11; // eax
  __int64 v12; // rdx
  char v13; // al
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  __int64 v16; // rax
  __int64 *v17; // rbx
  __int64 *v18; // r12
  __int64 v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // rdx
  unsigned __int64 v22; // r12
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // r15
  __int64 v26; // rbx
  unsigned __int64 v27; // r12
  unsigned __int64 v28; // r13
  unsigned __int64 v29; // rdi
  __int64 *v30; // rax
  __int64 v31; // rcx
  __int64 *v32; // rbx
  __int64 *v33; // r15
  __int64 v34; // rdi
  unsigned int v35; // ecx
  __int64 v36; // rsi
  __int64 *v37; // rbx
  __int64 v38; // rsi
  __int64 v39; // rdi
  __int64 *v41; // [rsp+8h] [rbp-38h]

  v2 = (__int64 *)a1[25];
  a1[25] = 0;
  if ( v2 )
  {
    sub_2E39BC0(v2);
    a2 = 8;
    j_j___libc_free_0((unsigned __int64)v2);
  }
  v3 = a1[26];
  a1[26] = 0;
  if ( v3 )
  {
    sub_301D560(v3);
    v4 = *(__int64 **)(v3 + 32);
    v41 = *(__int64 **)(v3 + 40);
    if ( v4 != v41 )
    {
      do
      {
        v5 = *v4;
        v6 = *(__int64 **)(*v4 + 16);
        if ( *(__int64 **)(*v4 + 8) == v6 )
        {
          *(_BYTE *)(v5 + 152) = 1;
        }
        else
        {
          v7 = *(__int64 **)(*v4 + 8);
          do
          {
            v8 = *v7++;
            sub_2EA4EF0(v8, a2);
          }
          while ( v6 != v7 );
          *(_BYTE *)(v5 + 152) = 1;
          v9 = *(_QWORD *)(v5 + 8);
          if ( v9 != *(_QWORD *)(v5 + 16) )
            *(_QWORD *)(v5 + 16) = v9;
        }
        v10 = *(_QWORD *)(v5 + 32);
        if ( v10 != *(_QWORD *)(v5 + 40) )
          *(_QWORD *)(v5 + 40) = v10;
        ++*(_QWORD *)(v5 + 56);
        if ( *(_BYTE *)(v5 + 84) )
        {
          *(_QWORD *)v5 = 0;
        }
        else
        {
          v11 = 4 * (*(_DWORD *)(v5 + 76) - *(_DWORD *)(v5 + 80));
          v12 = *(unsigned int *)(v5 + 72);
          if ( v11 < 0x20 )
            v11 = 32;
          if ( (unsigned int)v12 > v11 )
          {
            sub_C8C990(v5 + 56, a2);
          }
          else
          {
            a2 = 0xFFFFFFFFLL;
            memset(*(void **)(v5 + 64), -1, 8 * v12);
          }
          v13 = *(_BYTE *)(v5 + 84);
          *(_QWORD *)v5 = 0;
          if ( !v13 )
            _libc_free(*(_QWORD *)(v5 + 64));
        }
        v14 = *(_QWORD *)(v5 + 32);
        if ( v14 )
        {
          a2 = *(_QWORD *)(v5 + 48) - v14;
          j_j___libc_free_0(v14);
        }
        v15 = *(_QWORD *)(v5 + 8);
        if ( v15 )
        {
          a2 = *(_QWORD *)(v5 + 24) - v15;
          j_j___libc_free_0(v15);
        }
        ++v4;
      }
      while ( v41 != v4 );
      v16 = *(_QWORD *)(v3 + 32);
      if ( v16 != *(_QWORD *)(v3 + 40) )
        *(_QWORD *)(v3 + 40) = v16;
    }
    v17 = *(__int64 **)(v3 + 120);
    v18 = &v17[2 * *(unsigned int *)(v3 + 128)];
    while ( v18 != v17 )
    {
      v19 = v17[1];
      v20 = *v17;
      v17 += 2;
      sub_C7D6A0(v20, v19, 16);
    }
    *(_DWORD *)(v3 + 128) = 0;
    v21 = *(unsigned int *)(v3 + 80);
    if ( (_DWORD)v21 )
    {
      *(_QWORD *)(v3 + 136) = 0;
      v30 = *(__int64 **)(v3 + 72);
      v31 = *v30;
      v32 = &v30[v21];
      v33 = v30 + 1;
      *(_QWORD *)(v3 + 56) = *v30;
      for ( *(_QWORD *)(v3 + 64) = v31 + 4096; v32 != v33; v30 = *(__int64 **)(v3 + 72) )
      {
        v34 = *v33;
        v35 = (unsigned int)(v33 - v30) >> 7;
        v36 = 4096LL << v35;
        if ( v35 >= 0x1E )
          v36 = 0x40000000000LL;
        ++v33;
        sub_C7D6A0(v34, v36, 16);
      }
      *(_DWORD *)(v3 + 80) = 1;
      sub_C7D6A0(*v30, 4096, 16);
      v37 = *(__int64 **)(v3 + 120);
      v22 = (unsigned __int64)&v37[2 * *(unsigned int *)(v3 + 128)];
      if ( v37 == (__int64 *)v22 )
        goto LABEL_30;
      do
      {
        v38 = v37[1];
        v39 = *v37;
        v37 += 2;
        sub_C7D6A0(v39, v38, 16);
      }
      while ( (__int64 *)v22 != v37 );
    }
    v22 = *(_QWORD *)(v3 + 120);
LABEL_30:
    if ( v22 != v3 + 136 )
      _libc_free(v22);
    v23 = *(_QWORD *)(v3 + 72);
    if ( v23 != v3 + 88 )
      _libc_free(v23);
    v24 = *(_QWORD *)(v3 + 32);
    if ( v24 )
      j_j___libc_free_0(v24);
    sub_C7D6A0(*(_QWORD *)(v3 + 8), 16LL * *(unsigned int *)(v3 + 24), 8);
    j_j___libc_free_0(v3);
  }
  v25 = a1[27];
  a1[27] = 0;
  if ( v25 )
  {
    v26 = *(_QWORD *)(v25 + 24);
    v27 = v26 + 8LL * *(unsigned int *)(v25 + 32);
    if ( v26 != v27 )
    {
      do
      {
        v28 = *(_QWORD *)(v27 - 8);
        v27 -= 8LL;
        if ( v28 )
        {
          v29 = *(_QWORD *)(v28 + 24);
          if ( v29 != v28 + 40 )
            _libc_free(v29);
          j_j___libc_free_0(v28);
        }
      }
      while ( v26 != v27 );
      v27 = *(_QWORD *)(v25 + 24);
    }
    if ( v27 != v25 + 40 )
      _libc_free(v27);
    if ( *(_QWORD *)v25 != v25 + 16 )
      _libc_free(*(_QWORD *)v25);
    j_j___libc_free_0(v25);
  }
}
