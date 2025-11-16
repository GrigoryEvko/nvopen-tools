// Function: sub_130F1C0
// Address: 0x130f1c0
//
__int64 sub_130F1C0(__int64 a1, char *a2, ...)
{
  __int64 v2; // r8
  __int64 (__fastcall *v3)(__int64, _BYTE *); // rdi
  __int64 result; // rax
  gcc_va_list va; // [rsp+8h] [rbp-C8h] BYREF

  if ( *(_DWORD *)a1 == 2 )
  {
    v2 = *(_QWORD *)(a1 + 16);
    v3 = *(__int64 (__fastcall **)(__int64, _BYTE *))(a1 + 8);
    va_start(va, a2);
    return sub_40E25E(v3, v2, a2, (__int64)va);
  }
  return result;
}
