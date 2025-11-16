// Function: sub_C94DE0
// Address: 0xc94de0
//
int __fastcall sub_C94DE0(__int64 a1)
{
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = &unk_49DCB18;
  return pthread_key_create((pthread_key_t *)(a1 + 8), 0);
}
