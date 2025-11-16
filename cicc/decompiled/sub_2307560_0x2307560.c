// Function: sub_2307560
// Address: 0x2307560
//
void __fastcall sub_2307560(_QWORD *a1)
{
  _QWORD *v1; // r13
  _QWORD *v2; // rbx
  unsigned __int64 v3; // r12
  __int64 v4; // rdi

  v1 = a1 + 1;
  v2 = (_QWORD *)a1[1];
  *a1 = &unk_4A0E078;
  if ( v2 != a1 + 1 )
  {
    do
    {
      v3 = (unsigned __int64)v2;
      v2 = (_QWORD *)*v2;
      v4 = *(_QWORD *)(v3 + 16);
      if ( v4 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
      j_j___libc_free_0(v3);
    }
    while ( v2 != v1 );
  }
}
