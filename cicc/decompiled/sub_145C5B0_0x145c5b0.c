// Function: sub_145C5B0
// Address: 0x145c5b0
//
void *__fastcall sub_145C5B0(__int64 a1, _BYTE *a2, _BYTE *a3)
{
  size_t v3; // r15
  unsigned __int64 v5; // r13
  __int64 v6; // rdx
  void *result; // rax

  v3 = a3 - a2;
  v5 = (a3 - a2) >> 3;
  v6 = *(unsigned int *)(a1 + 8);
  result = (void *)(*(unsigned int *)(a1 + 12) - v6);
  if ( (unsigned __int64)result < v5 )
  {
    result = (void *)sub_16CD150(a1, a1 + 16, v5 + v6, 8);
    v6 = *(unsigned int *)(a1 + 8);
  }
  if ( a2 != a3 )
  {
    result = memcpy((void *)(*(_QWORD *)a1 + 8 * v6), a2, v3);
    LODWORD(v6) = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = v5 + v6;
  return result;
}
