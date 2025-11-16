// Function: sub_16E3A10
// Address: 0x16e3a10
//
__int64 __fastcall sub_16E3A10(__int64 a1)
{
  __int64 v2; // r12
  _QWORD *v3; // r14
  _QWORD *v4; // r12
  unsigned __int64 v5; // rdi
  __int64 v6; // rbx
  __int64 v7; // rbx
  __int64 v8; // r12
  unsigned __int64 v9; // r14
  __int64 v10; // rdi

  v2 = *(unsigned int *)(a1 + 56);
  v3 = *(_QWORD **)(a1 + 48);
  *(_QWORD *)a1 = &unk_49EF958;
  v4 = &v3[4 * v2];
  if ( v3 != v4 )
  {
    do
    {
      v4 -= 4;
      if ( (_QWORD *)*v4 != v4 + 2 )
        j_j___libc_free_0(*v4, v4[2] + 1LL);
    }
    while ( v3 != v4 );
    v4 = *(_QWORD **)(a1 + 48);
  }
  if ( v4 != (_QWORD *)(a1 + 64) )
    _libc_free((unsigned __int64)v4);
  v5 = *(_QWORD *)(a1 + 16);
  if ( *(_DWORD *)(a1 + 28) )
  {
    v6 = *(unsigned int *)(a1 + 24);
    if ( (_DWORD)v6 )
    {
      v7 = 8 * v6;
      v8 = 0;
      do
      {
        v9 = *(_QWORD *)(v5 + v8);
        if ( v9 != -8 && v9 )
        {
          v10 = *(_QWORD *)(v9 + 8);
          if ( v10 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 16LL))(v10);
          _libc_free(v9);
          v5 = *(_QWORD *)(a1 + 16);
        }
        v8 += 8;
      }
      while ( v7 != v8 );
    }
  }
  _libc_free(v5);
  return j_j___libc_free_0(a1, 256);
}
