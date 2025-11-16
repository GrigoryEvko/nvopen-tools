// Function: sub_1440B30
// Address: 0x1440b30
//
__int64 __fastcall sub_1440B30(__int64 a1)
{
  __int64 v2; // r13
  _QWORD *v3; // rbx
  _QWORD *v4; // r13
  __int64 v5; // r14
  __int64 v6; // rdi
  unsigned __int64 v7; // rdi

  v2 = *(unsigned int *)(a1 + 232);
  *(_QWORD *)a1 = &unk_49EB980;
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD **)(a1 + 216);
    v4 = &v3[2 * v2];
    do
    {
      if ( *v3 != -16 && *v3 != -8 )
      {
        v5 = v3[1];
        if ( v5 )
        {
          v6 = *(_QWORD *)(v5 + 24);
          if ( v6 )
            j_j___libc_free_0(v6, *(_QWORD *)(v5 + 40) - v6);
          j_j___libc_free_0(v5, 56);
        }
      }
      v3 += 2;
    }
    while ( v4 != v3 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 216));
  v7 = *(_QWORD *)(a1 + 160);
  if ( v7 != a1 + 176 )
    _libc_free(v7);
  *(_QWORD *)a1 = &unk_49EE078;
  sub_16366C0(a1);
  return j_j___libc_free_0(a1, 264);
}
