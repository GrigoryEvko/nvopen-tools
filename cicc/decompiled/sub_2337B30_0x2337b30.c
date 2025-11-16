// Function: sub_2337B30
// Address: 0x2337b30
//
void __fastcall sub_2337B30(unsigned __int64 *a1)
{
  _QWORD *v1; // r13
  _QWORD *v2; // r12
  _QWORD *v3; // r13
  _QWORD *v4; // r12

  v1 = (_QWORD *)a1[13];
  v2 = (_QWORD *)a1[12];
  if ( v1 != v2 )
  {
    do
    {
      if ( *v2 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v2 + 8LL))(*v2);
      ++v2;
    }
    while ( v1 != v2 );
    v2 = (_QWORD *)a1[12];
  }
  if ( v2 )
    j_j___libc_free_0((unsigned __int64)v2);
  v3 = (_QWORD *)a1[10];
  v4 = (_QWORD *)a1[9];
  if ( v3 != v4 )
  {
    do
    {
      if ( *v4 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v4 + 8LL))(*v4);
      ++v4;
    }
    while ( v3 != v4 );
    v4 = (_QWORD *)a1[9];
  }
  if ( v4 )
    j_j___libc_free_0((unsigned __int64)v4);
  if ( (unsigned __int64 *)*a1 != a1 + 2 )
    _libc_free(*a1);
}
