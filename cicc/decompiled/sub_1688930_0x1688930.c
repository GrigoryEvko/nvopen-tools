// Function: sub_1688930
// Address: 0x1688930
//
size_t __fastcall sub_1688930(__int64 a1, const char *a2, __m128i *a3)
{
  unsigned int v3; // eax
  size_t result; // rax
  __int64 *v5; // r13
  size_t v6; // r14
  __int64 v7; // rdx
  char *v8; // r13
  FILE *v9; // rdi
  __int64 v11; // [rsp+8h] [rbp-28h]

  if ( !a1 )
    return vfprintf(stdout, a2, a3);
  v3 = *(_DWORD *)a1;
  if ( *(_DWORD *)a1 == 3 )
  {
    v9 = *(FILE **)(a1 + 32);
    if ( v9 )
      return vfprintf(v9, a2, a3);
    return vfprintf(stdout, a2, a3);
  }
  if ( v3 > 3 )
  {
    if ( v3 == 4 )
    {
      result = vsprintf(*(char **)(a1 + 32), a2, a3);
      *(_QWORD *)(a1 + 32) += (int)result;
    }
    else
    {
      return -1;
    }
  }
  else if ( v3 == 2 )
  {
    return sub_1688540(*(__int64 **)(a1 + 32), a2, a3);
  }
  else
  {
    v5 = sub_1688290(128, (__int64)a2, (__int64)a3);
    sub_1688540(v5, a2, a3);
    v6 = sub_16886C0((__int64)v5);
    v8 = sub_16884C0(v5, (__int64)a2, v7);
    v11 = sub_1688820(a1, v8, v6);
    sub_16856A0(v8);
    return v11;
  }
  return result;
}
