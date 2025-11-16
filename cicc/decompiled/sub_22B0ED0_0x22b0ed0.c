// Function: sub_22B0ED0
// Address: 0x22b0ed0
//
__int64 __fastcall sub_22B0ED0(__int64 a1)
{
  unsigned __int64 v1; // r13
  __int64 v2; // rsi
  __int64 *v3; // r14
  __int64 *v4; // rbx
  __int64 i; // rax
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 v8; // rsi
  __int64 *v9; // rbx
  unsigned __int64 v10; // r12
  __int64 v11; // rsi
  __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 *v14; // r14
  __int64 *v15; // rbx
  __int64 j; // rax
  __int64 v17; // rdi
  unsigned int v18; // ecx
  __int64 v19; // rsi
  __int64 *v20; // rbx
  unsigned __int64 v21; // r12
  __int64 v22; // rsi
  __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  unsigned __int64 *v26; // r15
  unsigned __int64 *v27; // r14
  unsigned __int64 v28; // rbx
  unsigned __int64 v29; // r12
  __int64 v30; // rsi
  __int64 v31; // rdi

  v1 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = 0;
  if ( v1 )
  {
    if ( *(_BYTE *)(v1 + 336) )
    {
      v26 = *(unsigned __int64 **)(v1 + 320);
      v27 = *(unsigned __int64 **)(v1 + 312);
      *(_BYTE *)(v1 + 336) = 0;
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
        while ( v26 != v27 );
        v27 = *(unsigned __int64 **)(v1 + 312);
      }
      if ( v27 )
        j_j___libc_free_0((unsigned __int64)v27);
    }
    sub_C7D6A0(*(_QWORD *)(v1 + 240), 16LL * *(unsigned int *)(v1 + 256), 8);
    v2 = 16LL * *(unsigned int *)(v1 + 224);
    if ( !*(_DWORD *)(v1 + 224) )
      v2 = 0;
    sub_C7D6A0(*(_QWORD *)(v1 + 208), v2, 8);
    sub_E66D20(v1 + 96);
    v3 = *(__int64 **)(v1 + 112);
    v4 = &v3[*(unsigned int *)(v1 + 120)];
    if ( v3 != v4 )
    {
      for ( i = *(_QWORD *)(v1 + 112); ; i = *(_QWORD *)(v1 + 112) )
      {
        v6 = *v3;
        v7 = (unsigned int)(((__int64)v3 - i) >> 3) >> 7;
        v8 = 4096LL << v7;
        if ( v7 >= 0x1E )
          v8 = 0x40000000000LL;
        ++v3;
        sub_C7D6A0(v6, v8, 16);
        if ( v4 == v3 )
          break;
      }
    }
    v9 = *(__int64 **)(v1 + 160);
    v10 = (unsigned __int64)&v9[2 * *(unsigned int *)(v1 + 168)];
    if ( v9 != (__int64 *)v10 )
    {
      do
      {
        v11 = v9[1];
        v12 = *v9;
        v9 += 2;
        sub_C7D6A0(v12, v11, 16);
      }
      while ( (__int64 *)v10 != v9 );
      v10 = *(_QWORD *)(v1 + 160);
    }
    if ( v10 != v1 + 176 )
      _libc_free(v10);
    v13 = *(_QWORD *)(v1 + 112);
    if ( v13 != v1 + 128 )
      _libc_free(v13);
    sub_22B0CF0(v1);
    v14 = *(__int64 **)(v1 + 16);
    v15 = &v14[*(unsigned int *)(v1 + 24)];
    if ( v14 != v15 )
    {
      for ( j = *(_QWORD *)(v1 + 16); ; j = *(_QWORD *)(v1 + 16) )
      {
        v17 = *v14;
        v18 = (unsigned int)(((__int64)v14 - j) >> 3) >> 7;
        v19 = 4096LL << v18;
        if ( v18 >= 0x1E )
          v19 = 0x40000000000LL;
        ++v14;
        sub_C7D6A0(v17, v19, 16);
        if ( v15 == v14 )
          break;
      }
    }
    v20 = *(__int64 **)(v1 + 64);
    v21 = (unsigned __int64)&v20[2 * *(unsigned int *)(v1 + 72)];
    if ( v20 != (__int64 *)v21 )
    {
      do
      {
        v22 = v20[1];
        v23 = *v20;
        v20 += 2;
        sub_C7D6A0(v23, v22, 16);
      }
      while ( (__int64 *)v21 != v20 );
      v21 = *(_QWORD *)(v1 + 64);
    }
    if ( v21 != v1 + 80 )
      _libc_free(v21);
    v24 = *(_QWORD *)(v1 + 16);
    if ( v24 != v1 + 32 )
      _libc_free(v24);
    j_j___libc_free_0(v1);
  }
  return 0;
}
