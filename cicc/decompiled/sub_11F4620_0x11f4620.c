// Function: sub_11F4620
// Address: 0x11f4620
//
__int64 *sub_11F4620(
        __int64 *a1,
        __int64 (__fastcall *a2)(_BYTE *, __int64, __int64, __va_list_tag *),
        __int64 a3,
        __int64 a4,
        ...)
{
  void *v4; // rsp
  int v5; // eax
  _BYTE v7[8]; // [rsp+0h] [rbp-E0h] BYREF
  gcc_va_list va; // [rsp+8h] [rbp-D8h] BYREF

  va_start(va, a4);
  v4 = alloca(a3 + 8);
  v5 = a2(v7, a3, a4, va);
  *a1 = (__int64)(a1 + 2);
  sub_11F4570(a1, v7, (__int64)&v7[v5]);
  return a1;
}
