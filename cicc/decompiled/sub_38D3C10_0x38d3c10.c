// Function: sub_38D3C10
// Address: 0x38d3c10
//
__int64 __fastcall sub_38D3C10(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // r13
  char *v5; // [rsp+0h] [rbp-30h] BYREF
  char v6; // [rsp+10h] [rbp-20h]
  char v7; // [rsp+11h] [rbp-1Fh]

  v2 = a1[1];
  v7 = 1;
  v5 = "cfi";
  v6 = 3;
  v3 = sub_38BF8E0(v2, (__int64)&v5, 1, 1);
  (*(void (__fastcall **)(_QWORD *, __int64, _QWORD))(*a1 + 176LL))(a1, v3, 0);
  return v3;
}
