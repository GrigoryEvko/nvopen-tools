// Function: sub_233A900
// Address: 0x233a900
//
void __fastcall sub_233A900(_QWORD *a1)
{
  _QWORD *v1; // r13
  _QWORD *v2; // r12
  _QWORD *v3; // r13
  _QWORD *v4; // r12
  _QWORD *v5; // r13
  _QWORD *v6; // r12

  v1 = (_QWORD *)a1[24];
  v2 = (_QWORD *)a1[23];
  if ( v1 != v2 )
  {
    do
    {
      if ( *v2 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v2 + 8LL))(*v2);
      ++v2;
    }
    while ( v1 != v2 );
    v2 = (_QWORD *)a1[23];
  }
  if ( v2 )
    j_j___libc_free_0((unsigned __int64)v2);
  v3 = (_QWORD *)a1[19];
  v4 = (_QWORD *)a1[18];
  if ( v3 != v4 )
  {
    do
    {
      if ( *v4 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v4 + 8LL))(*v4);
      ++v4;
    }
    while ( v3 != v4 );
    v4 = (_QWORD *)a1[18];
  }
  if ( v4 )
    j_j___libc_free_0((unsigned __int64)v4);
  v5 = (_QWORD *)a1[14];
  v6 = (_QWORD *)a1[13];
  if ( v5 != v6 )
  {
    do
    {
      if ( *v6 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v6 + 8LL))(*v6);
      ++v6;
    }
    while ( v5 != v6 );
    v6 = (_QWORD *)a1[13];
  }
  if ( v6 )
    j_j___libc_free_0((unsigned __int64)v6);
}
