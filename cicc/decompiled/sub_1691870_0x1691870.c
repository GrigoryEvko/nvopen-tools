// Function: sub_1691870
// Address: 0x1691870
//
void __fastcall sub_1691870(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  unsigned __int64 v4; // r8
  __int64 v5; // r13
  __int64 v6; // r13
  __int64 v7; // rbx
  unsigned __int64 v8; // rdi

  v2 = a1 + 88;
  v3 = *(_QWORD *)(a1 + 72);
  if ( v3 != v2 )
    j_j___libc_free_0(v3, *(_QWORD *)(a1 + 88) + 1LL);
  v4 = *(_QWORD *)(a1 + 40);
  if ( *(_DWORD *)(a1 + 52) )
  {
    v5 = *(unsigned int *)(a1 + 48);
    if ( (_DWORD)v5 )
    {
      v6 = 8 * v5;
      v7 = 0;
      do
      {
        v8 = *(_QWORD *)(v4 + v7);
        if ( v8 && v8 != -8 )
        {
          _libc_free(v8);
          v4 = *(_QWORD *)(a1 + 40);
        }
        v7 += 8;
      }
      while ( v6 != v7 );
    }
  }
  _libc_free(v4);
  if ( *(_QWORD *)a1 )
    j_j___libc_free_0(*(_QWORD *)a1, *(_QWORD *)(a1 + 16) - *(_QWORD *)a1);
}
