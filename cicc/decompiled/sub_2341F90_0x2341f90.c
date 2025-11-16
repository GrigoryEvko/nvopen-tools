// Function: sub_2341F90
// Address: 0x2341f90
//
__int64 __fastcall sub_2341F90(__int64 a1, __int64 a2)
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
  unsigned int v22; // ecx
  unsigned int v23; // eax
  _QWORD *v24; // rdi
  int v25; // r12d
  _QWORD *v26; // rax
  unsigned int v27; // eax
  _QWORD *v28; // rax
  __int64 v29; // rdx
  _QWORD *j; // rdx
  __int64 *v31; // [rsp+8h] [rbp-38h]

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
  v22 = 4 * v3;
  a2 = 64;
  v4 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)(4 * v3) < 0x40 )
    v22 = 64;
  if ( v22 >= (unsigned int)v4 )
  {
LABEL_4:
    v5 = *(_QWORD **)(a1 + 16);
    for ( i = &v5[2 * v4]; i != v5; v5 += 2 )
      *v5 = -4096;
    goto LABEL_6;
  }
  v23 = v3 - 1;
  if ( v23 )
  {
    _BitScanReverse(&v23, v23);
    v24 = *(_QWORD **)(a1 + 16);
    v25 = 1 << (33 - (v23 ^ 0x1F));
    if ( v25 < 64 )
      v25 = 64;
    if ( v25 == (_DWORD)v4 )
    {
      *(_QWORD *)(a1 + 24) = 0;
      v26 = &v24[2 * (unsigned int)v25];
      do
      {
        if ( v24 )
          *v24 = -4096;
        v24 += 2;
      }
      while ( v26 != v24 );
      goto LABEL_7;
    }
  }
  else
  {
    v24 = *(_QWORD **)(a1 + 16);
    v25 = 64;
  }
  a2 = 16 * v4;
  sub_C7D6A0((__int64)v24, 16 * v4, 8);
  v27 = sub_2309150(v25);
  *(_DWORD *)(a1 + 32) = v27;
  if ( !v27 )
    goto LABEL_48;
  a2 = 8;
  v28 = (_QWORD *)sub_C7D670(16LL * v27, 8);
  v29 = *(unsigned int *)(a1 + 32);
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = v28;
  for ( j = &v28[2 * v29]; j != v28; v28 += 2 )
  {
    if ( v28 )
      *v28 = -4096;
  }
LABEL_7:
  v7 = *(__int64 **)(a1 + 40);
  v31 = *(__int64 **)(a1 + 48);
  if ( v7 != v31 )
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
    while ( v31 != v7 );
    v19 = *(_QWORD *)(a1 + 40);
    if ( v19 != *(_QWORD *)(a1 + 48) )
      *(_QWORD *)(a1 + 48) = v19;
  }
  sub_E66D20(a1 + 64);
  sub_B72320(a1 + 64, a2);
  v20 = *(_QWORD *)(a1 + 40);
  if ( v20 )
    j_j___libc_free_0(v20);
  return sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * *(unsigned int *)(a1 + 32), 8);
}
