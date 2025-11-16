// Function: sub_1BBA200
// Address: 0x1bba200
//
__int64 __fastcall sub_1BBA200(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 *v7; // r15
  __int64 *v8; // r13
  __int64 v9; // r12
  __int64 v10; // rbx
  unsigned __int64 v11; // rdi

  v2 = a1 + 120;
  v3 = *(_QWORD *)(a1 + 104);
  if ( v3 != v2 )
    _libc_free(v3);
  v4 = *(unsigned int *)(a1 + 96);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 80);
    v6 = v5 + 88 * v4;
    do
    {
      if ( *(_QWORD *)v5 != -16 && *(_QWORD *)v5 != -8 && (*(_BYTE *)(v5 + 16) & 1) == 0 )
        j___libc_free_0(*(_QWORD *)(v5 + 24));
      v5 += 88;
    }
    while ( v6 != v5 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 80));
  j___libc_free_0(*(_QWORD *)(a1 + 48));
  v7 = *(__int64 **)(a1 + 16);
  v8 = *(__int64 **)(a1 + 8);
  if ( v7 != v8 )
  {
    do
    {
      v9 = *v8;
      if ( *v8 )
      {
        v10 = v9 + 112LL * *(_QWORD *)(v9 - 8);
        while ( v9 != v10 )
        {
          v10 -= 112;
          v11 = *(_QWORD *)(v10 + 32);
          if ( v11 != v10 + 48 )
            _libc_free(v11);
        }
        j_j_j___libc_free_0_0(v9 - 8);
      }
      ++v8;
    }
    while ( v7 != v8 );
    v8 = *(__int64 **)(a1 + 8);
  }
  if ( v8 )
    j_j___libc_free_0(v8, *(_QWORD *)(a1 + 24) - (_QWORD)v8);
  return j_j___libc_free_0(a1, 232);
}
