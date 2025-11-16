// Function: sub_134CA00
// Address: 0x134ca00
//
__int64 __fastcall sub_134CA00(_QWORD *a1)
{
  __int64 v2; // rdi
  _QWORD *v3; // rbx
  _QWORD *v4; // r12

  v2 = a1[9];
  if ( v2 )
    j_j___libc_free_0(v2, a1[11] - v2);
  v3 = (_QWORD *)a1[7];
  v4 = (_QWORD *)a1[6];
  if ( v3 != v4 )
  {
    do
    {
      if ( *v4 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v4 + 8LL))(*v4);
      ++v4;
    }
    while ( v3 != v4 );
    v4 = (_QWORD *)a1[6];
  }
  if ( v4 )
    j_j___libc_free_0(v4, a1[8] - (_QWORD)v4);
  return j___libc_free_0(a1[2]);
}
