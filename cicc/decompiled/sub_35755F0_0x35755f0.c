// Function: sub_35755F0
// Address: 0x35755f0
//
void __fastcall sub_35755F0(_QWORD *a1)
{
  unsigned __int64 v1; // rsi

  v1 = a1[25];
  *a1 = &unk_4A394F0;
  if ( v1 )
    sub_3575560((__int64)(a1 + 25), v1);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
