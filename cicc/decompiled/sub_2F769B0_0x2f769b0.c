// Function: sub_2F769B0
// Address: 0x2f769b0
//
void *__fastcall sub_2F769B0(__int64 a1, unsigned int a2)
{
  void *v3; // rdi
  void *result; // rax

  *(_DWORD *)(a1 + 8) = a2;
  v3 = *(void **)a1;
  if ( *(_DWORD *)(a1 + 12) >= a2 )
    return memset(v3, 0, (unsigned __int64)a2 << 6);
  *(_DWORD *)(a1 + 12) = a2;
  _libc_free((unsigned __int64)v3);
  result = _libc_calloc(a2, 0x40u);
  if ( !result && (a2 || (result = (void *)malloc(1u)) == 0) )
    sub_C64F00("Allocation failed", 1u);
  *(_QWORD *)a1 = result;
  return result;
}
