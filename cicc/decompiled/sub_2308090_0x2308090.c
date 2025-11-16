// Function: sub_2308090
// Address: 0x2308090
//
void __fastcall sub_2308090(_QWORD *a1)
{
  __int64 v2; // rdi

  *a1 = &unk_4A0D978;
  v2 = a1[1];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  j_j___libc_free_0((unsigned __int64)a1);
}
