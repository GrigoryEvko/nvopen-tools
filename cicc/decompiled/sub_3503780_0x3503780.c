// Function: sub_3503780
// Address: 0x3503780
//
__int64 __fastcall sub_3503780(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r15
  __int64 v3; // rbx
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // r13
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // r13
  __int64 *v8; // r12
  __int64 v9; // r15
  __int64 *v10; // rbx
  __int64 *v11; // r14
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned int v15; // eax
  __int64 v16; // rdx
  char v17; // al
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  __int64 v20; // rax
  __int64 *v21; // rbx
  __int64 *v22; // r12
  __int64 v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rdx
  unsigned __int64 v26; // r12
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // r12
  __int64 *v31; // rax
  __int64 v32; // rcx
  __int64 *v33; // rbx
  __int64 *v34; // r15
  __int64 v35; // rdi
  unsigned int v36; // ecx
  __int64 v37; // rsi
  __int64 *v38; // rbx
  __int64 v39; // rsi
  __int64 v40; // rdi
  __int64 *v42; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 216);
  *(_QWORD *)a1 = &unk_4A38828;
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 24);
    v4 = v3 + 8LL * *(unsigned int *)(v2 + 32);
    if ( v3 != v4 )
    {
      do
      {
        v5 = *(_QWORD *)(v4 - 8);
        v4 -= 8LL;
        if ( v5 )
        {
          v6 = *(_QWORD *)(v5 + 24);
          if ( v6 != v5 + 40 )
            _libc_free(v6);
          j_j___libc_free_0(v5);
        }
      }
      while ( v3 != v4 );
      v4 = *(_QWORD *)(v2 + 24);
    }
    if ( v4 != v2 + 40 )
      _libc_free(v4);
    if ( *(_QWORD *)v2 != v2 + 16 )
      _libc_free(*(_QWORD *)v2);
    a2 = 128;
    j_j___libc_free_0(v2);
  }
  v7 = *(_QWORD *)(a1 + 208);
  if ( v7 )
  {
    sub_301D560(*(_QWORD *)(a1 + 208));
    v8 = *(__int64 **)(v7 + 32);
    v42 = *(__int64 **)(v7 + 40);
    if ( v8 != v42 )
    {
      do
      {
        v9 = *v8;
        v10 = *(__int64 **)(*v8 + 16);
        if ( *(__int64 **)(*v8 + 8) == v10 )
        {
          *(_BYTE *)(v9 + 152) = 1;
        }
        else
        {
          v11 = *(__int64 **)(*v8 + 8);
          do
          {
            v12 = *v11++;
            sub_2EA4EF0(v12, a2);
          }
          while ( v10 != v11 );
          *(_BYTE *)(v9 + 152) = 1;
          v13 = *(_QWORD *)(v9 + 8);
          if ( v13 != *(_QWORD *)(v9 + 16) )
            *(_QWORD *)(v9 + 16) = v13;
        }
        v14 = *(_QWORD *)(v9 + 32);
        if ( v14 != *(_QWORD *)(v9 + 40) )
          *(_QWORD *)(v9 + 40) = v14;
        ++*(_QWORD *)(v9 + 56);
        if ( *(_BYTE *)(v9 + 84) )
        {
          *(_QWORD *)v9 = 0;
        }
        else
        {
          v15 = 4 * (*(_DWORD *)(v9 + 76) - *(_DWORD *)(v9 + 80));
          v16 = *(unsigned int *)(v9 + 72);
          if ( v15 < 0x20 )
            v15 = 32;
          if ( (unsigned int)v16 > v15 )
          {
            sub_C8C990(v9 + 56, a2);
          }
          else
          {
            a2 = 0xFFFFFFFFLL;
            memset(*(void **)(v9 + 64), -1, 8 * v16);
          }
          v17 = *(_BYTE *)(v9 + 84);
          *(_QWORD *)v9 = 0;
          if ( !v17 )
            _libc_free(*(_QWORD *)(v9 + 64));
        }
        v18 = *(_QWORD *)(v9 + 32);
        if ( v18 )
        {
          a2 = *(_QWORD *)(v9 + 48) - v18;
          j_j___libc_free_0(v18);
        }
        v19 = *(_QWORD *)(v9 + 8);
        if ( v19 )
        {
          a2 = *(_QWORD *)(v9 + 24) - v19;
          j_j___libc_free_0(v19);
        }
        ++v8;
      }
      while ( v42 != v8 );
      v20 = *(_QWORD *)(v7 + 32);
      if ( v20 != *(_QWORD *)(v7 + 40) )
        *(_QWORD *)(v7 + 40) = v20;
    }
    v21 = *(__int64 **)(v7 + 120);
    v22 = &v21[2 * *(unsigned int *)(v7 + 128)];
    while ( v22 != v21 )
    {
      v23 = v21[1];
      v24 = *v21;
      v21 += 2;
      sub_C7D6A0(v24, v23, 16);
    }
    *(_DWORD *)(v7 + 128) = 0;
    v25 = *(unsigned int *)(v7 + 80);
    if ( (_DWORD)v25 )
    {
      *(_QWORD *)(v7 + 136) = 0;
      v31 = *(__int64 **)(v7 + 72);
      v32 = *v31;
      v33 = &v31[v25];
      v34 = v31 + 1;
      *(_QWORD *)(v7 + 56) = *v31;
      for ( *(_QWORD *)(v7 + 64) = v32 + 4096; v33 != v34; v31 = *(__int64 **)(v7 + 72) )
      {
        v35 = *v34;
        v36 = (unsigned int)(v34 - v31) >> 7;
        v37 = 4096LL << v36;
        if ( v36 >= 0x1E )
          v37 = 0x40000000000LL;
        ++v34;
        sub_C7D6A0(v35, v37, 16);
      }
      *(_DWORD *)(v7 + 80) = 1;
      sub_C7D6A0(*v31, 4096, 16);
      v38 = *(__int64 **)(v7 + 120);
      v26 = (unsigned __int64)&v38[2 * *(unsigned int *)(v7 + 128)];
      if ( v38 == (__int64 *)v26 )
        goto LABEL_41;
      do
      {
        v39 = v38[1];
        v40 = *v38;
        v38 += 2;
        sub_C7D6A0(v40, v39, 16);
      }
      while ( (__int64 *)v26 != v38 );
    }
    v26 = *(_QWORD *)(v7 + 120);
LABEL_41:
    if ( v26 != v7 + 136 )
      _libc_free(v26);
    v27 = *(_QWORD *)(v7 + 72);
    if ( v27 != v7 + 88 )
      _libc_free(v27);
    v28 = *(_QWORD *)(v7 + 32);
    if ( v28 )
      j_j___libc_free_0(v28);
    sub_C7D6A0(*(_QWORD *)(v7 + 8), 16LL * *(unsigned int *)(v7 + 24), 8);
    j_j___libc_free_0(v7);
  }
  v29 = *(_QWORD *)(a1 + 200);
  if ( v29 )
  {
    sub_2E39BC0(*(__int64 **)(a1 + 200));
    j_j___libc_free_0(v29);
  }
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
