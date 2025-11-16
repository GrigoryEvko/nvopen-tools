// Function: sub_163A210
// Address: 0x163a210
//
__int64 __fastcall sub_163A210(__int64 a1)
{
  __int64 v2; // rdi
  __int64 *v3; // rbx
  __int64 *v4; // r12
  __int64 v5; // r13
  __int64 v6; // rdi
  unsigned __int64 v7; // r8
  __int64 v8; // rbx
  __int64 v9; // rbx
  __int64 v10; // r12
  unsigned __int64 v11; // rdi

  v2 = *(_QWORD *)(a1 + 104);
  if ( v2 )
    j_j___libc_free_0(v2, *(_QWORD *)(a1 + 120) - v2);
  v3 = *(__int64 **)(a1 + 88);
  v4 = *(__int64 **)(a1 + 80);
  if ( v3 != v4 )
  {
    do
    {
      v5 = *v4;
      if ( *v4 )
      {
        v6 = *(_QWORD *)(v5 + 48);
        if ( v6 )
          j_j___libc_free_0(v6, *(_QWORD *)(v5 + 64) - v6);
        j_j___libc_free_0(v5, 80);
      }
      ++v4;
    }
    while ( v3 != v4 );
    v4 = *(__int64 **)(a1 + 80);
  }
  if ( v4 )
    j_j___libc_free_0(v4, *(_QWORD *)(a1 + 96) - (_QWORD)v4);
  v7 = *(_QWORD *)(a1 + 48);
  if ( *(_DWORD *)(a1 + 60) )
  {
    v8 = *(unsigned int *)(a1 + 56);
    if ( (_DWORD)v8 )
    {
      v9 = 8 * v8;
      v10 = 0;
      do
      {
        v11 = *(_QWORD *)(v7 + v10);
        if ( v11 && v11 != -8 )
        {
          _libc_free(v11);
          v7 = *(_QWORD *)(a1 + 48);
        }
        v10 += 8;
      }
      while ( v9 != v10 );
    }
  }
  _libc_free(v7);
  j___libc_free_0(*(_QWORD *)(a1 + 24));
  return sub_16C9010(a1);
}
