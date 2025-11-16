// Function: sub_CD07E0
// Address: 0xcd07e0
//
__int64 __fastcall sub_CD07E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v8; // rax
  __int64 v9; // rbx
  _QWORD *v10; // rdx
  __int64 v11; // r12
  __int64 *v12; // r15
  __int64 *v13; // rbx
  __int64 i; // rax
  __int64 v15; // rdi
  unsigned int v16; // ecx
  __int64 *v17; // rbx
  __int64 *v18; // r14
  __int64 v19; // rdi
  __int64 v20; // rdi

  v8 = sub_22077B0(152);
  v9 = v8;
  if ( v8 )
    sub_CCFFF0(v8, (__int64 *)a2, a3, a4, a5, 0, 0);
  *(_QWORD *)a1 = v9;
  v10 = (_QWORD *)sub_22077B0(96);
  if ( v10 )
  {
    memset(v10, 0, 0x60u);
    v10[11] = 1;
    v10[2] = v10 + 4;
    v10[3] = 0x400000000LL;
    v10[8] = v10 + 10;
  }
  v11 = *(_QWORD *)(*(_QWORD *)a1 + 144LL);
  *(_QWORD *)(*(_QWORD *)a1 + 144LL) = v10;
  if ( v11 )
  {
    v12 = *(__int64 **)(v11 + 16);
    v13 = &v12[*(unsigned int *)(v11 + 24)];
    if ( v12 != v13 )
    {
      for ( i = *(_QWORD *)(v11 + 16); ; i = *(_QWORD *)(v11 + 16) )
      {
        v15 = *v12;
        v16 = (unsigned int)(((__int64)v12 - i) >> 3) >> 7;
        a2 = 4096LL << v16;
        if ( v16 >= 0x1E )
          a2 = 0x40000000000LL;
        ++v12;
        sub_C7D6A0(v15, a2, 16);
        if ( v13 == v12 )
          break;
      }
    }
    v17 = *(__int64 **)(v11 + 64);
    v18 = &v17[2 * *(unsigned int *)(v11 + 72)];
    if ( v17 != v18 )
    {
      do
      {
        a2 = v17[1];
        v19 = *v17;
        v17 += 2;
        sub_C7D6A0(v19, a2, 16);
      }
      while ( v18 != v17 );
      v18 = *(__int64 **)(v11 + 64);
    }
    if ( v18 != (__int64 *)(v11 + 80) )
      _libc_free(v18, a2);
    v20 = *(_QWORD *)(v11 + 16);
    if ( v20 != v11 + 32 )
      _libc_free(v20, a2);
    j_j___libc_free_0(v11, 96);
  }
  return a1;
}
