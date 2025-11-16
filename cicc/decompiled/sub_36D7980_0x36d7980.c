// Function: sub_36D7980
// Address: 0x36d7980
//
void __fastcall sub_36D7980(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_4A36A08;
  v2 = a1[25];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
