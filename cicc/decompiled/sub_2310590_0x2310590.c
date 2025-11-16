// Function: sub_2310590
// Address: 0x2310590
//
void __fastcall sub_2310590(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  __int64 v3; // rdi

  v1 = (_QWORD *)a1[3];
  v2 = (_QWORD *)a1[2];
  *a1 = &unk_4A11E38;
  if ( v1 != v2 )
  {
    do
    {
      if ( *v2 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v2 + 8LL))(*v2);
      ++v2;
    }
    while ( v1 != v2 );
    v2 = (_QWORD *)a1[2];
  }
  if ( v2 )
    j_j___libc_free_0((unsigned __int64)v2);
  v3 = a1[1];
  if ( v3 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
}
