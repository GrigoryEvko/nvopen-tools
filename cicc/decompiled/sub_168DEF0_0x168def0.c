// Function: sub_168DEF0
// Address: 0x168def0
//
__int64 __fastcall sub_168DEF0(__int64 a1)
{
  __int64 v2; // r13
  _QWORD *v3; // rbx
  _QWORD *v4; // r13
  __int64 v5; // rdi
  __int64 v6; // r13
  unsigned __int64 v7; // r8
  __int64 v8; // r13
  __int64 v9; // rbx
  unsigned __int64 v10; // rdi

  v2 = *(unsigned int *)(a1 + 328);
  *(_QWORD *)a1 = &unk_49EE5B0;
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 312);
    v4 = &v3[4 * v2];
    do
    {
      if ( *v3 != -16 && *v3 != -8 )
      {
        v5 = v3[1];
        if ( v5 )
          j_j___libc_free_0(v5, v3[3] - v5);
      }
      v3 += 4;
    }
    while ( v4 != v3 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 312));
  if ( *(_DWORD *)(a1 + 284) )
  {
    v6 = *(unsigned int *)(a1 + 280);
    v7 = *(_QWORD *)(a1 + 272);
    if ( (_DWORD)v6 )
    {
      v8 = 8 * v6;
      v9 = 0;
      do
      {
        v10 = *(_QWORD *)(v7 + v9);
        if ( v10 && v10 != -8 )
        {
          _libc_free(v10);
          v7 = *(_QWORD *)(a1 + 272);
        }
        v9 += 8;
      }
      while ( v8 != v9 );
    }
  }
  else
  {
    v7 = *(_QWORD *)(a1 + 272);
  }
  _libc_free(v7);
  sub_38DCBC0(a1);
  return j_j___libc_free_0(a1, 336);
}
