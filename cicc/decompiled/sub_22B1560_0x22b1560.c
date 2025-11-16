// Function: sub_22B1560
// Address: 0x22b1560
//
void __fastcall sub_22B1560(_QWORD *a1)
{
  unsigned __int64 v2; // r12
  __int64 v3; // rsi
  __int64 *v4; // r15
  __int64 *v5; // rbx
  __int64 i; // rax
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 v9; // rsi
  __int64 *v10; // rbx
  unsigned __int64 v11; // r13
  __int64 v12; // rsi
  __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  __int64 *v15; // r15
  __int64 *v16; // rbx
  __int64 j; // rax
  __int64 v18; // rdi
  unsigned int v19; // ecx
  __int64 v20; // rsi
  __int64 *v21; // rbx
  unsigned __int64 v22; // r13
  __int64 v23; // rsi
  __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  unsigned __int64 *v26; // rax
  unsigned __int64 *v27; // r13
  unsigned __int64 v28; // rbx
  unsigned __int64 v29; // r15
  __int64 v30; // rsi
  __int64 v31; // rdi
  unsigned __int64 *v32; // [rsp+8h] [rbp-38h]

  v2 = a1[22];
  *a1 = &unk_4A09CE8;
  if ( v2 )
  {
    if ( *(_BYTE *)(v2 + 336) )
    {
      v26 = *(unsigned __int64 **)(v2 + 320);
      v27 = *(unsigned __int64 **)(v2 + 312);
      *(_BYTE *)(v2 + 336) = 0;
      v32 = v26;
      if ( v26 != v27 )
      {
        do
        {
          v28 = v27[1];
          v29 = *v27;
          if ( v28 != *v27 )
          {
            do
            {
              v30 = *(unsigned int *)(v29 + 144);
              v31 = *(_QWORD *)(v29 + 128);
              v29 += 152LL;
              sub_C7D6A0(v31, 8 * v30, 4);
              sub_C7D6A0(*(_QWORD *)(v29 - 56), 8LL * *(unsigned int *)(v29 - 40), 4);
              sub_C7D6A0(*(_QWORD *)(v29 - 88), 16LL * *(unsigned int *)(v29 - 72), 8);
              sub_C7D6A0(*(_QWORD *)(v29 - 120), 16LL * *(unsigned int *)(v29 - 104), 8);
            }
            while ( v28 != v29 );
            v29 = *v27;
          }
          if ( v29 )
            j_j___libc_free_0(v29);
          v27 += 3;
        }
        while ( v32 != v27 );
        v27 = *(unsigned __int64 **)(v2 + 312);
      }
      if ( v27 )
        j_j___libc_free_0((unsigned __int64)v27);
    }
    sub_C7D6A0(*(_QWORD *)(v2 + 240), 16LL * *(unsigned int *)(v2 + 256), 8);
    v3 = 16LL * *(unsigned int *)(v2 + 224);
    if ( !*(_DWORD *)(v2 + 224) )
      v3 = 0;
    sub_C7D6A0(*(_QWORD *)(v2 + 208), v3, 8);
    sub_E66D20(v2 + 96);
    v4 = *(__int64 **)(v2 + 112);
    v5 = &v4[*(unsigned int *)(v2 + 120)];
    if ( v4 != v5 )
    {
      for ( i = *(_QWORD *)(v2 + 112); ; i = *(_QWORD *)(v2 + 112) )
      {
        v7 = *v4;
        v8 = (unsigned int)(((__int64)v4 - i) >> 3) >> 7;
        v9 = 4096LL << v8;
        if ( v8 >= 0x1E )
          v9 = 0x40000000000LL;
        ++v4;
        sub_C7D6A0(v7, v9, 16);
        if ( v5 == v4 )
          break;
      }
    }
    v10 = *(__int64 **)(v2 + 160);
    v11 = (unsigned __int64)&v10[2 * *(unsigned int *)(v2 + 168)];
    if ( v10 != (__int64 *)v11 )
    {
      do
      {
        v12 = v10[1];
        v13 = *v10;
        v10 += 2;
        sub_C7D6A0(v13, v12, 16);
      }
      while ( (__int64 *)v11 != v10 );
      v11 = *(_QWORD *)(v2 + 160);
    }
    if ( v11 != v2 + 176 )
      _libc_free(v11);
    v14 = *(_QWORD *)(v2 + 112);
    if ( v14 != v2 + 128 )
      _libc_free(v14);
    sub_22B0CF0(v2);
    v15 = *(__int64 **)(v2 + 16);
    v16 = &v15[*(unsigned int *)(v2 + 24)];
    if ( v15 != v16 )
    {
      for ( j = *(_QWORD *)(v2 + 16); ; j = *(_QWORD *)(v2 + 16) )
      {
        v18 = *v15;
        v19 = (unsigned int)(((__int64)v15 - j) >> 3) >> 7;
        v20 = 4096LL << v19;
        if ( v19 >= 0x1E )
          v20 = 0x40000000000LL;
        ++v15;
        sub_C7D6A0(v18, v20, 16);
        if ( v16 == v15 )
          break;
      }
    }
    v21 = *(__int64 **)(v2 + 64);
    v22 = (unsigned __int64)&v21[2 * *(unsigned int *)(v2 + 72)];
    if ( v21 != (__int64 *)v22 )
    {
      do
      {
        v23 = v21[1];
        v24 = *v21;
        v21 += 2;
        sub_C7D6A0(v24, v23, 16);
      }
      while ( (__int64 *)v22 != v21 );
      v22 = *(_QWORD *)(v2 + 64);
    }
    if ( v22 != v2 + 80 )
      _libc_free(v22);
    v25 = *(_QWORD *)(v2 + 16);
    if ( v25 != v2 + 32 )
      _libc_free(v25);
    j_j___libc_free_0(v2);
  }
  sub_BB9260((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
