// Function: sub_C20520
// Address: 0xc20520
//
__int64 __fastcall sub_C20520(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  __int64 v6; // rdi

  *a1 = &unk_49DBCC8;
  v3 = a1[34];
  if ( v3 )
  {
    a2 = a1[36] - v3;
    j_j___libc_free_0(v3, a2);
  }
  v4 = (_QWORD *)a1[32];
  v5 = (_QWORD *)a1[31];
  if ( v4 != v5 )
  {
    do
    {
      if ( (_QWORD *)*v5 != v5 + 2 )
        _libc_free(*v5, a2);
      v5 += 5;
    }
    while ( v4 != v5 );
    v5 = (_QWORD *)a1[31];
  }
  if ( v5 )
    j_j___libc_free_0(v5, a1[33] - (_QWORD)v5);
  v6 = a1[28];
  if ( v6 )
    j_j___libc_free_0(v6, a1[30] - v6);
  return sub_C201C0((__int64)a1);
}
