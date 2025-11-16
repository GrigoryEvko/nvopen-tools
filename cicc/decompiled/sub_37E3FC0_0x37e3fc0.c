// Function: sub_37E3FC0
// Address: 0x37e3fc0
//
void __fastcall sub_37E3FC0(__int64 **a1, __int64 a2)
{
  __int64 *v2; // r15
  __int64 v3; // r13
  _DWORD *v4; // r14
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 v7; // rcx
  __m128i *v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // r9
  _DWORD *v12; // rax
  __int64 *v13; // r12
  __int64 v14; // rbx
  unsigned __int64 v15; // rdi
  __int64 v16; // [rsp+8h] [rbp-38h]

  v2 = *a1;
  v3 = 107LL * *((unsigned int *)a1 + 2);
  v16 = (__int64)&(*a1)[v3];
  if ( v2 != &v2[v3] )
  {
    v4 = (_DWORD *)(a2 + 824);
    v5 = a2 + 88;
    do
    {
      v6 = v5 - 88;
      if ( v5 != 88 )
      {
        v7 = *v2;
        *(_QWORD *)(v5 - 80) = 0;
        v8 = (__m128i *)(v5 - 64);
        *(_QWORD *)(v5 - 88) = v7;
        *(_DWORD *)(v6 + 16) = 1;
        *(_DWORD *)(v5 - 68) = 0;
        do
        {
          if ( v8 )
            v8->m128i_i32[0] = -1;
          v8 = (__m128i *)((char *)v8 + 8);
        }
        while ( (__m128i *)v5 != v8 );
        sub_37E3D10((_DWORD *)(v5 - 80), (_DWORD *)v2 + 2);
        *(_DWORD *)(v5 + 8) = 0;
        *(_QWORD *)v5 = v5 + 16;
        *(_DWORD *)(v5 + 12) = 8;
        if ( *((_DWORD *)v2 + 24) )
          sub_37B73F0(v5, (__int64)(v2 + 11), v9, v5 + 16, v10, v11);
        v8[37].m128i_i64[0] = 0;
        *(_DWORD *)(v6 + 688) = 1;
        v12 = (_DWORD *)(v5 + 608);
        v8[37].m128i_i32[3] = 0;
        do
        {
          if ( v12 )
            *v12 = -1;
          v12 += 4;
        }
        while ( v12 != v4 );
        sub_37E3E60((_DWORD *)(v5 + 592), (_DWORD *)v2 + 170);
        v8[46].m128i_i64[0] = v2[103];
        v8[46].m128i_i64[1] = v2[104];
        v8[47] = _mm_loadu_si128((const __m128i *)(v2 + 105));
      }
      v2 += 107;
      v4 += 214;
      v5 += 856;
    }
    while ( (__int64 *)v16 != v2 );
    v13 = *a1;
    v14 = (__int64)&(*a1)[107 * *((unsigned int *)a1 + 2)];
    while ( (__int64 *)v14 != v13 )
    {
      while ( 1 )
      {
        v14 -= 856;
        if ( (*(_BYTE *)(v14 + 688) & 1) == 0 )
          sub_C7D6A0(*(_QWORD *)(v14 + 696), 16LL * *(unsigned int *)(v14 + 704), 8);
        v15 = *(_QWORD *)(v14 + 88);
        if ( v15 != v14 + 104 )
          _libc_free(v15);
        if ( (*(_BYTE *)(v14 + 16) & 1) != 0 )
          break;
        sub_C7D6A0(*(_QWORD *)(v14 + 24), 8LL * *(unsigned int *)(v14 + 32), 4);
        if ( (__int64 *)v14 == v13 )
          return;
      }
    }
  }
}
