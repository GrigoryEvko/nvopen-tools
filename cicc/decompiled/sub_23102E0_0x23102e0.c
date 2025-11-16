// Function: sub_23102E0
// Address: 0x23102e0
//
void __fastcall sub_23102E0(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12

  v1 = (_QWORD *)a1[2];
  v2 = (_QWORD *)a1[1];
  *a1 = &unk_4A0C438;
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
    j_j___libc_free_0((unsigned __int64)v2);
}
