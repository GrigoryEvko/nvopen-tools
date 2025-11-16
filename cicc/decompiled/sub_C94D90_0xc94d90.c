// Function: sub_C94D90
// Address: 0xc94d90
//
int __fastcall sub_C94D90(__int64 a1)
{
  *(_QWORD *)a1 = &unk_49DCB18;
  return pthread_key_delete(*(_DWORD *)(a1 + 8));
}
