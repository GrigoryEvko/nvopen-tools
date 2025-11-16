// Function: sub_2307660
// Address: 0x2307660
//
void __fastcall sub_2307660(_QWORD *a1)
{
  _QWORD *v2; // r13
  _QWORD *v3; // rbx
  unsigned __int64 v4; // r12
  __int64 v5; // rdi

  v2 = a1 + 1;
  v3 = (_QWORD *)a1[1];
  *a1 = &unk_4A0E078;
  if ( v3 != a1 + 1 )
  {
    do
    {
      v4 = (unsigned __int64)v3;
      v3 = (_QWORD *)*v3;
      v5 = *(_QWORD *)(v4 + 16);
      if ( v5 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
      j_j___libc_free_0(v4);
    }
    while ( v3 != v2 );
  }
  j_j___libc_free_0((unsigned __int64)a1);
}
