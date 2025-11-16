// Function: sub_160FDE0
// Address: 0x160fde0
//
void __fastcall sub_160FDE0(_QWORD *a1)
{
  __int64 v1; // rdi

  *a1 = &unk_49EDBC8;
  v1 = a1[1];
  if ( v1 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v1 + 8LL))(v1);
  nullsub_569();
}
