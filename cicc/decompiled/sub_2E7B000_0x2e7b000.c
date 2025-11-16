// Function: sub_2E7B000
// Address: 0x2e7b000
//
void *__fastcall sub_2E7B000(__int64 a1)
{
  int v1; // edx
  __int64 v2; // rax
  __int64 v3; // rdx
  unsigned __int64 v4; // rcx
  void *v6; // rax
  size_t n; // [rsp+8h] [rbp-18h]

  v1 = *(_DWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 16) + 200LL))(*(_QWORD *)(a1 + 16)) + 16);
  v2 = *(_QWORD *)(a1 + 128);
  v3 = 4LL * ((unsigned int)(v1 + 31) >> 5);
  *(_QWORD *)(a1 + 208) += v3;
  v4 = v3 + ((v2 + 3) & 0xFFFFFFFFFFFFFFFCLL);
  if ( *(_QWORD *)(a1 + 136) >= v4 && v2 )
  {
    *(_QWORD *)(a1 + 128) = v4;
    return memset((void *)((v2 + 3) & 0xFFFFFFFFFFFFFFFCLL), 0, v3);
  }
  else
  {
    n = v3;
    v6 = (void *)sub_9D1E70(a1 + 128, v3, v3, 2);
    return memset(v6, 0, n);
  }
}
