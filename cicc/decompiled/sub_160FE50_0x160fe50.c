// Function: sub_160FE50
// Address: 0x160fe50
//
void __fastcall sub_160FE50(_QWORD *a1)
{
  __int64 v1; // rdi

  *a1 = &unk_49EDE08;
  v1 = a1[2];
  if ( v1 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v1 + 8LL))(v1);
  nullsub_569();
}
