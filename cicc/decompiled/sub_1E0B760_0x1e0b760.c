// Function: sub_1E0B760
// Address: 0x1e0b760
//
void *__fastcall sub_1E0B760(__int64 a1)
{
  __int64 (*v1)(); // rax
  __int64 v2; // r12
  void *v3; // rax

  v1 = *(__int64 (**)())(**(_QWORD **)(a1 + 16) + 112LL);
  if ( v1 == sub_1D00B10 )
    BUG();
  v2 = 4LL * ((unsigned int)(*(_DWORD *)(v1() + 16) + 31) >> 5);
  v3 = (void *)sub_145CBF0((__int64 *)(a1 + 120), v2, 4);
  return memset(v3, 0, v2);
}
