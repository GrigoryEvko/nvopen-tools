// Function: sub_25B64D0
// Address: 0x25b64d0
//
void __fastcall sub_25B64D0(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = off_4A1F200;
  v2 = a1[22];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 56LL))(v2);
  sub_BB9260((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
