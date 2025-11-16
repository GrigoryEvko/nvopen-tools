// Function: sub_2305960
// Address: 0x2305960
//
void __fastcall sub_2305960(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_4A0ACA0;
  v2 = a1[1];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  j_j___libc_free_0((unsigned __int64)a1);
}
