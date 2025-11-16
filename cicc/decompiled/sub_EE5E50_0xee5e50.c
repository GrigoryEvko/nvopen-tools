// Function: sub_EE5E50
// Address: 0xee5e50
//
void __fastcall sub_EE5E50(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 *v3; // r14
  __int64 *v4; // rbx
  __int64 i; // rax
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 *v8; // rbx
  __int64 *v9; // r13
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rdi

  v2 = *a1;
  if ( *a1 )
  {
    if ( (*(_BYTE *)(v2 + 952) & 1) == 0 )
    {
      a2 = 16LL * *(unsigned int *)(v2 + 968);
      sub_C7D6A0(*(_QWORD *)(v2 + 960), a2, 8);
    }
    sub_C65770((_QWORD *)(v2 + 904), a2);
    v3 = *(__int64 **)(v2 + 824);
    v4 = &v3[*(unsigned int *)(v2 + 832)];
    if ( v3 != v4 )
    {
      for ( i = *(_QWORD *)(v2 + 824); ; i = *(_QWORD *)(v2 + 824) )
      {
        v6 = *v3;
        v7 = (unsigned int)(((__int64)v3 - i) >> 3) >> 7;
        a2 = 4096LL << v7;
        if ( v7 >= 0x1E )
          a2 = 0x40000000000LL;
        ++v3;
        sub_C7D6A0(v6, a2, 16);
        if ( v4 == v3 )
          break;
      }
    }
    v8 = *(__int64 **)(v2 + 872);
    v9 = &v8[2 * *(unsigned int *)(v2 + 880)];
    if ( v8 != v9 )
    {
      do
      {
        a2 = v8[1];
        v10 = *v8;
        v8 += 2;
        sub_C7D6A0(v10, a2, 16);
      }
      while ( v9 != v8 );
      v9 = *(__int64 **)(v2 + 872);
    }
    if ( v9 != (__int64 *)(v2 + 888) )
      _libc_free(v9, a2);
    v11 = *(_QWORD *)(v2 + 824);
    if ( v11 != v2 + 840 )
      _libc_free(v11, a2);
    v12 = *(_QWORD *)(v2 + 720);
    if ( v12 != v2 + 744 )
      _libc_free(v12, a2);
    v13 = *(_QWORD *)(v2 + 664);
    if ( v13 != v2 + 688 )
      _libc_free(v13, a2);
    v14 = *(_QWORD *)(v2 + 576);
    if ( v14 != v2 + 600 )
      _libc_free(v14, a2);
    v15 = *(_QWORD *)(v2 + 296);
    if ( v15 != v2 + 320 )
      _libc_free(v15, a2);
    v16 = *(_QWORD *)(v2 + 16);
    if ( v16 != v2 + 40 )
      _libc_free(v16, a2);
    j_j___libc_free_0(v2, 1472);
  }
}
