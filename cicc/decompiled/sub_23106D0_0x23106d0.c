// Function: sub_23106D0
// Address: 0x23106d0
//
void __fastcall sub_23106D0(_QWORD *a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  __int64 v4; // rdi

  v2 = (_QWORD *)a1[3];
  v3 = (_QWORD *)a1[2];
  *a1 = &unk_4A11E38;
  if ( v2 != v3 )
  {
    do
    {
      if ( *v3 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v3 + 8LL))(*v3);
      ++v3;
    }
    while ( v2 != v3 );
    v3 = (_QWORD *)a1[2];
  }
  if ( v3 )
    j_j___libc_free_0((unsigned __int64)v3);
  v4 = a1[1];
  if ( v4 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
  j_j___libc_free_0((unsigned __int64)a1);
}
