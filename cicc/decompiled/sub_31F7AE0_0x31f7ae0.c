// Function: sub_31F7AE0
// Address: 0x31f7ae0
//
__int64 __fastcall sub_31F7AE0(__int64 a1)
{
  __int64 *v2; // rdi
  __int64 v3; // rax
  void (*v4)(); // rax
  const char *v6; // [rsp+0h] [rbp-40h] BYREF
  char v7; // [rsp+20h] [rbp-20h]
  char v8; // [rsp+21h] [rbp-1Fh]

  (*(void (__fastcall **)(_QWORD, __int64, _QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 528) + 608LL))(
    *(_QWORD *)(a1 + 528),
    2,
    0,
    1,
    0);
  v2 = *(__int64 **)(a1 + 528);
  v3 = *v2;
  v8 = 1;
  v6 = "Debug section magic";
  v4 = *(void (**)())(v3 + 120);
  v7 = 3;
  if ( v4 != nullsub_98 )
  {
    ((void (__fastcall *)(__int64 *, const char **, __int64))v4)(v2, &v6, 1);
    v2 = *(__int64 **)(a1 + 528);
  }
  return (*(__int64 (__fastcall **)(__int64 *, __int64, __int64))(*v2 + 536))(v2, 4, 4);
}
