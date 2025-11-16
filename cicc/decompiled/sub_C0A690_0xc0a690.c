// Function: sub_C0A690
// Address: 0xc0a690
//
__int64 __fastcall sub_C0A690(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax
  void *v4; // rdi
  unsigned int v5; // r13d
  size_t v6; // rdx

  result = 0x800000000LL;
  v4 = a1 + 2;
  *a1 = v4;
  a1[1] = 0x800000000LL;
  v5 = *(_DWORD *)(a2 + 8);
  if ( v5 && a1 != (_QWORD *)a2 )
  {
    v6 = 16LL * v5;
    if ( v5 <= 8
      || (result = sub_C8D5F0(a1, v4, v5, 16), v4 = (void *)*a1, (v6 = 16LL * *(unsigned int *)(a2 + 8)) != 0) )
    {
      result = (__int64)memcpy(v4, *(const void **)a2, v6);
    }
    *((_DWORD *)a1 + 2) = v5;
  }
  return result;
}
