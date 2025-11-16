// Function: sub_CA8840
// Address: 0xca8840
//
void __fastcall sub_CA8840(__int64 *a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rdi
  __int64 *v6; // r15
  __int64 *v7; // rbx
  __int64 i; // rax
  __int64 v9; // rdi
  unsigned int v10; // ecx
  __int64 *v11; // rbx
  __int64 *v12; // r14
  __int64 v13; // rdi
  __int64 v14; // rdi
  __int64 v15; // r12

  v3 = a1[1];
  if ( v3 )
  {
    v4 = *(_QWORD *)(v3 + 128);
    while ( v4 )
    {
      sub_CA65A0(*(_QWORD *)(v4 + 24));
      v5 = v4;
      v4 = *(_QWORD *)(v4 + 16);
      a2 = 64;
      j_j___libc_free_0(v5, 64);
    }
    v6 = *(__int64 **)(v3 + 24);
    v7 = &v6[*(unsigned int *)(v3 + 32)];
    if ( v6 != v7 )
    {
      for ( i = *(_QWORD *)(v3 + 24); ; i = *(_QWORD *)(v3 + 24) )
      {
        v9 = *v6;
        v10 = (unsigned int)(((__int64)v6 - i) >> 3) >> 7;
        a2 = 4096LL << v10;
        if ( v10 >= 0x1E )
          a2 = 0x40000000000LL;
        ++v6;
        sub_C7D6A0(v9, a2, 16);
        if ( v7 == v6 )
          break;
      }
    }
    v11 = *(__int64 **)(v3 + 72);
    v12 = &v11[2 * *(unsigned int *)(v3 + 80)];
    if ( v11 != v12 )
    {
      do
      {
        a2 = v11[1];
        v13 = *v11;
        v11 += 2;
        sub_C7D6A0(v13, a2, 16);
      }
      while ( v12 != v11 );
      v12 = *(__int64 **)(v3 + 72);
    }
    if ( v12 != (__int64 *)(v3 + 88) )
      _libc_free(v12, a2);
    v14 = *(_QWORD *)(v3 + 24);
    if ( v14 != v3 + 40 )
      _libc_free(v14, a2);
    a2 = 160;
    j_j___libc_free_0(v3, 160);
  }
  v15 = *a1;
  if ( *a1 )
  {
    sub_CA6E30(*a1, a2);
    j_j___libc_free_0(v15, 344);
  }
}
