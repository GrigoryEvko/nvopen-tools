// Function: sub_CF4BF0
// Address: 0xcf4bf0
//
__int64 __fastcall sub_CF4BF0(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 result; // rax
  _QWORD *v4; // rbx
  _QWORD *v5; // r12

  v2 = a1[4];
  if ( v2 )
    result = j_j___libc_free_0(v2, a1[6] - v2);
  v4 = (_QWORD *)a1[2];
  v5 = (_QWORD *)a1[1];
  if ( v4 != v5 )
  {
    do
    {
      if ( *v5 )
        result = (*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*v5 + 8LL))(*v5);
      ++v5;
    }
    while ( v4 != v5 );
    v5 = (_QWORD *)a1[1];
  }
  if ( v5 )
    return j_j___libc_free_0(v5, a1[3] - (_QWORD)v5);
  return result;
}
