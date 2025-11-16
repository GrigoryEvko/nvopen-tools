// Function: sub_130F0B0
// Address: 0x130f0b0
//
__int64 sub_130F0B0(__int64 a1, char *a2, ...)
{
  gcc_va_list va; // [rsp+8h] [rbp-C8h] BYREF

  va_start(va, a2);
  return sub_40E25E(*(__int64 (__fastcall **)(__int64, _BYTE *))(a1 + 8), *(_QWORD *)(a1 + 16), a2, (__int64)va);
}
