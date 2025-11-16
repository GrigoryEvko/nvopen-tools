// Function: sub_341F1D0
// Address: 0x341f1d0
//
void __fastcall sub_341F1D0(_QWORD *a1)
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
