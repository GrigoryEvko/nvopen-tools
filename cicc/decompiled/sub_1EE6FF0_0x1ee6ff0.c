// Function: sub_1EE6FF0
// Address: 0x1ee6ff0
//
void *__fastcall sub_1EE6FF0(__int64 a1, unsigned int a2)
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
  if ( !result )
  {
    if ( a2 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      result = 0;
    }
    else
    {
      result = (void *)sub_13A3880(1u);
    }
  }
  *(_QWORD *)a1 = result;
  return result;
}
