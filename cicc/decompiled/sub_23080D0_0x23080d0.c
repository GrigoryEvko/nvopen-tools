// Function: sub_23080D0
// Address: 0x23080d0
//
void __fastcall sub_23080D0(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_4A0B3D0;
  v2 = a1[3];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  j_j___libc_free_0((unsigned __int64)a1);
}
