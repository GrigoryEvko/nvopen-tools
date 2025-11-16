// Function: sub_F94A20
// Address: 0xf94a20
//
void __fastcall sub_F94A20(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // rdi

  nullsub_61();
  v2 = a1 + 2;
  a1[16] = &unk_49DA100;
  nullsub_63();
  v3 = (_QWORD *)*a1;
  if ( v3 != v2 )
    _libc_free(v3, a2);
}
