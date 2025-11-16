// Function: sub_C20430
// Address: 0xc20430
//
__int64 __fastcall sub_C20430(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // r13
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  _QWORD *v6; // rdi

  v3 = a1 + 26;
  v4 = (_QWORD *)a1[26];
  *a1 = &unk_49DBAF0;
  if ( a1 + 26 != v4 )
  {
    do
    {
      v5 = v4;
      v4 = (_QWORD *)*v4;
      v6 = (_QWORD *)v5[2];
      if ( v6 != v5 + 4 )
        _libc_free(v6, a2);
      a2 = 56;
      j_j___libc_free_0(v5, 56);
    }
    while ( v3 != v4 );
  }
  return sub_C201C0((__int64)a1);
}
