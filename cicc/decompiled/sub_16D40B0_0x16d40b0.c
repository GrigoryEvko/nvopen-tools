// Function: sub_16D40B0
// Address: 0x16d40b0
//
int __fastcall sub_16D40B0(__int64 a1)
{
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = &unk_49EF4C8;
  return pthread_key_create((pthread_key_t *)(a1 + 8), 0);
}
