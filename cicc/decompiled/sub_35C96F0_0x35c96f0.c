// Function: sub_35C96F0
// Address: 0x35c96f0
//
void *__fastcall sub_35C96F0(__int64 a1)
{
  void *v2; // rdi
  void *v3; // rdi
  void *result; // rax
  __int64 v5; // rax
  __int64 v6; // rax

  *(_DWORD *)(a1 + 44) = 0;
  v2 = *(void **)(a1 + 72);
  if ( !v2 )
  {
    *(_QWORD *)(a1 + 80) = 1;
    v5 = sub_2207820(8u);
    *(_QWORD *)(a1 + 72) = v5;
    v2 = (void *)v5;
  }
  memset(v2, 0, 8LL * *(_QWORD *)(a1 + 80));
  v3 = *(void **)(a1 + 48);
  *(_QWORD *)(a1 + 88) = 0;
  if ( !v3 )
  {
    *(_QWORD *)(a1 + 56) = 1;
    v6 = sub_2207820(8u);
    *(_QWORD *)(a1 + 48) = v6;
    v3 = (void *)v6;
  }
  result = memset(v3, 0, 8LL * *(_QWORD *)(a1 + 56));
  *(_QWORD *)(a1 + 64) = 0;
  return result;
}
