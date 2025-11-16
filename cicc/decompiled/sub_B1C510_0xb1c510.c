// Function: sub_B1C510
// Address: 0xb1c510
//
__int64 __fastcall sub_B1C510(_QWORD *a1, const void *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // r13
  size_t v6; // r12
  void *v8; // rdi

  result = 0x4000000000LL;
  v4 = 16 * a3;
  v5 = v4 >> 4;
  v6 = v4;
  v8 = a1 + 2;
  *a1 = v8;
  a1[1] = 0x4000000000LL;
  if ( (unsigned __int64)v4 > 0x400 )
  {
    sub_C8D5F0(a1, v8, v4 >> 4, 16);
    v8 = (void *)(*a1 + 16LL * *((unsigned int *)a1 + 2));
  }
  else if ( !v4 )
  {
    *((_DWORD *)a1 + 2) = v5;
    return result;
  }
  result = (__int64)memcpy(v8, a2, v6);
  *((_DWORD *)a1 + 2) += v5;
  return result;
}
