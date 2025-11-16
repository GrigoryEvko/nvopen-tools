// Function: sub_12F0990
// Address: 0x12f0990
//
__int64 __fastcall sub_12F0990(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdi
  _QWORD *v4; // rbx
  _QWORD *v5; // r12
  __int64 v6; // rdi

  *a1 = &unk_49E75F8;
  v3 = a1[23];
  if ( v3 )
  {
    a2 = a1[25] - v3;
    j_j___libc_free_0(v3, a2);
  }
  v4 = (_QWORD *)a1[21];
  v5 = (_QWORD *)a1[20];
  if ( v4 != v5 )
  {
    do
    {
      if ( (_QWORD *)*v5 != v5 + 2 )
      {
        a2 = v5[2] + 1LL;
        j_j___libc_free_0(*v5, a2);
      }
      v5 += 4;
    }
    while ( v4 != v5 );
    v5 = (_QWORD *)a1[20];
  }
  if ( v5 )
  {
    a2 = a1[22] - (_QWORD)v5;
    j_j___libc_free_0(v5, a2);
  }
  v6 = a1[12];
  if ( v6 != a1[11] )
    _libc_free(v6, a2);
  return j_j___libc_free_0(a1, 216);
}
