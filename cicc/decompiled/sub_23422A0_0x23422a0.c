// Function: sub_23422A0
// Address: 0x23422a0
//
void __fastcall sub_23422A0(unsigned __int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rdx
  _QWORD *v5; // rax
  _QWORD *i; // rdx
  __int64 *v7; // r12
  __int64 v8; // r15
  __int64 *v9; // rbx
  __int64 *v10; // r14
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned int v14; // eax
  __int64 v15; // rdx
  char v16; // al
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  unsigned __int64 v20; // rdi
  unsigned int v21; // ecx
  unsigned int v22; // eax
  _QWORD *v23; // rdi
  int v24; // r12d
  _QWORD *v25; // rax
  unsigned int v26; // eax
  _QWORD *v27; // rax
  __int64 v28; // rdx
  _QWORD *j; // rdx
  __int64 *v30; // [rsp+8h] [rbp-38h]

  ++*(_QWORD *)(a1 + 8);
  *(_QWORD *)a1 = &unk_4A15908;
  v3 = *(_DWORD *)(a1 + 24);
  if ( !v3 )
  {
    if ( !*(_DWORD *)(a1 + 28) )
      goto LABEL_7;
    v4 = *(unsigned int *)(a1 + 32);
    if ( (unsigned int)v4 <= 0x40 )
      goto LABEL_4;
    a2 = 16 * v4;
    sub_C7D6A0(*(_QWORD *)(a1 + 16), 16 * v4, 8);
    *(_DWORD *)(a1 + 32) = 0;
LABEL_48:
    *(_QWORD *)(a1 + 16) = 0;
LABEL_6:
    *(_QWORD *)(a1 + 24) = 0;
    goto LABEL_7;
  }
  v21 = 4 * v3;
  a2 = 64;
  v4 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)(4 * v3) < 0x40 )
    v21 = 64;
  if ( v21 >= (unsigned int)v4 )
  {
LABEL_4:
    v5 = *(_QWORD **)(a1 + 16);
    for ( i = &v5[2 * v4]; i != v5; v5 += 2 )
      *v5 = -4096;
    goto LABEL_6;
  }
  v22 = v3 - 1;
  if ( v22 )
  {
    _BitScanReverse(&v22, v22);
    v23 = *(_QWORD **)(a1 + 16);
    v24 = 1 << (33 - (v22 ^ 0x1F));
    if ( v24 < 64 )
      v24 = 64;
    if ( v24 == (_DWORD)v4 )
    {
      *(_QWORD *)(a1 + 24) = 0;
      v25 = &v23[2 * (unsigned int)v24];
      do
      {
        if ( v23 )
          *v23 = -4096;
        v23 += 2;
      }
      while ( v25 != v23 );
      goto LABEL_7;
    }
  }
  else
  {
    v23 = *(_QWORD **)(a1 + 16);
    v24 = 64;
  }
  a2 = 16 * v4;
  sub_C7D6A0((__int64)v23, 16 * v4, 8);
  v26 = sub_2309150(v24);
  *(_DWORD *)(a1 + 32) = v26;
  if ( !v26 )
    goto LABEL_48;
  a2 = 8;
  v27 = (_QWORD *)sub_C7D670(16LL * v26, 8);
  v28 = *(unsigned int *)(a1 + 32);
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = v27;
  for ( j = &v27[2 * v28]; j != v27; v27 += 2 )
  {
    if ( v27 )
      *v27 = -4096;
  }
LABEL_7:
  v7 = *(__int64 **)(a1 + 40);
  v30 = *(__int64 **)(a1 + 48);
  if ( v7 != v30 )
  {
    do
    {
      v8 = *v7;
      v9 = *(__int64 **)(*v7 + 16);
      if ( *(__int64 **)(*v7 + 8) == v9 )
      {
        *(_BYTE *)(v8 + 152) = 1;
      }
      else
      {
        v10 = *(__int64 **)(*v7 + 8);
        do
        {
          v11 = *v10++;
          sub_D47BB0(v11, a2);
        }
        while ( v9 != v10 );
        *(_BYTE *)(v8 + 152) = 1;
        v12 = *(_QWORD *)(v8 + 8);
        if ( v12 != *(_QWORD *)(v8 + 16) )
          *(_QWORD *)(v8 + 16) = v12;
      }
      v13 = *(_QWORD *)(v8 + 32);
      if ( v13 != *(_QWORD *)(v8 + 40) )
        *(_QWORD *)(v8 + 40) = v13;
      ++*(_QWORD *)(v8 + 56);
      if ( *(_BYTE *)(v8 + 84) )
      {
        *(_QWORD *)v8 = 0;
      }
      else
      {
        v14 = 4 * (*(_DWORD *)(v8 + 76) - *(_DWORD *)(v8 + 80));
        v15 = *(unsigned int *)(v8 + 72);
        if ( v14 < 0x20 )
          v14 = 32;
        if ( v14 < (unsigned int)v15 )
        {
          sub_C8C990(v8 + 56, a2);
        }
        else
        {
          a2 = 0xFFFFFFFFLL;
          memset(*(void **)(v8 + 64), -1, 8 * v15);
        }
        v16 = *(_BYTE *)(v8 + 84);
        *(_QWORD *)v8 = 0;
        if ( !v16 )
          _libc_free(*(_QWORD *)(v8 + 64));
      }
      v17 = *(_QWORD *)(v8 + 32);
      if ( v17 )
      {
        a2 = *(_QWORD *)(v8 + 48) - v17;
        j_j___libc_free_0(v17);
      }
      v18 = *(_QWORD *)(v8 + 8);
      if ( v18 )
      {
        a2 = *(_QWORD *)(v8 + 24) - v18;
        j_j___libc_free_0(v18);
      }
      ++v7;
    }
    while ( v30 != v7 );
    v19 = *(_QWORD *)(a1 + 40);
    if ( v19 != *(_QWORD *)(a1 + 48) )
      *(_QWORD *)(a1 + 48) = v19;
  }
  sub_E66D20(a1 + 64);
  sub_B72320(a1 + 64, a2);
  v20 = *(_QWORD *)(a1 + 40);
  if ( v20 )
    j_j___libc_free_0(v20);
  sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * *(unsigned int *)(a1 + 32), 8);
  j_j___libc_free_0(a1);
}
