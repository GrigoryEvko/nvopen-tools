// Function: sub_CB0A00
// Address: 0xcb0a00
//
void __fastcall sub_CB0A00(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // rdi

  *a1 = &unk_49DCF98;
  v3 = (_QWORD *)a1[4];
  if ( v3 != a1 + 6 )
    _libc_free(v3, a2);
  nullsub_175();
}
