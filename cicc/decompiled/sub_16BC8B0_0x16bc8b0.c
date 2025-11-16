// Function: sub_16BC8B0
// Address: 0x16bc8b0
//
__int64 __fastcall sub_16BC8B0(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12

  v1 = (_QWORD *)a1[2];
  v2 = (_QWORD *)a1[1];
  *a1 = &unk_49EF270;
  if ( v1 != v2 )
  {
    do
    {
      if ( *v2 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v2 + 8LL))(*v2);
      ++v2;
    }
    while ( v1 != v2 );
    v2 = (_QWORD *)a1[1];
  }
  if ( v2 )
    j_j___libc_free_0(v2, a1[3] - (_QWORD)v2);
  return j_j___libc_free_0(a1, 32);
}
