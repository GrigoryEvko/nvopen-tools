// Function: sub_3363A30
// Address: 0x3363a30
//
void __fastcall sub_3363A30(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  *a1 = off_4A36640;
  v2 = a1[83];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = a1[79];
  if ( v3 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 16LL))(v3);
  v4 = a1[80];
  if ( v4 )
    j_j___libc_free_0(v4);
  v5 = a1[76];
  *a1 = &unk_4A365B8;
  if ( v5 )
    j_j___libc_free_0(v5);
  sub_2F8EAD0(a1);
}
