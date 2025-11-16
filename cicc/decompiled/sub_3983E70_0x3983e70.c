// Function: sub_3983E70
// Address: 0x3983e70
//
__int64 __fastcall sub_3983E70(__int64 a1, __int64 a2)
{
  __int64 v2; // rdi
  __int64 (__fastcall *v3)(__int64, __int64, __int64 **); // rax
  __int64 v5; // [rsp+8h] [rbp-28h] BYREF
  __int64 *v6; // [rsp+10h] [rbp-20h] BYREF
  __int16 v7; // [rsp+20h] [rbp-10h]

  v2 = *(_QWORD *)(a1 + 88);
  v5 = a2;
  v3 = *(__int64 (__fastcall **)(__int64, __int64, __int64 **))(*(_QWORD *)v2 + 16LL);
  v7 = 267;
  v6 = &v5;
  return v3(v2, a2, &v6);
}
