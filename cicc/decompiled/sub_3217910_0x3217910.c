// Function: sub_3217910
// Address: 0x3217910
//
__int64 __fastcall sub_3217910(__int64 a1, __int64 a2)
{
  bool v2; // zf
  __int64 v3; // rdi
  __int64 (__fastcall *v4)(__int64, __int64, __int64 **, __int64); // rax
  __int64 v6; // [rsp+8h] [rbp-38h] BYREF
  __int64 *v7; // [rsp+10h] [rbp-30h] BYREF
  __int16 v8; // [rsp+30h] [rbp-10h]

  v2 = *(_BYTE *)(a1 + 120) == 0;
  v6 = a2;
  if ( v2 )
    v3 = *(_QWORD *)(a1 + 112);
  else
    v3 = *(_QWORD *)(a1 + 104) + 80LL;
  v4 = *(__int64 (__fastcall **)(__int64, __int64, __int64 **, __int64))(*(_QWORD *)v3 + 16LL);
  v8 = 267;
  v7 = &v6;
  return v4(v3, v6, &v7, 4);
}
