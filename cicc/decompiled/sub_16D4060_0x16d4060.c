// Function: sub_16D4060
// Address: 0x16d4060
//
int __fastcall sub_16D4060(__int64 a1)
{
  *(_QWORD *)a1 = &unk_49EF4C8;
  return pthread_key_delete(*(_DWORD *)(a1 + 8));
}
