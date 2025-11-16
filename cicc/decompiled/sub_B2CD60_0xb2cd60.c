// Function: sub_B2CD60
// Address: 0xb2cd60
//
__int64 __fastcall sub_B2CD60(__int64 a1, const void *a2, size_t a3, const void *a4, size_t a5)
{
  _QWORD *v8; // rax
  __int64 v9; // r9
  __int64 v11; // [rsp-10h] [rbp-40h]

  v8 = (_QWORD *)sub_B2BE50(a1);
  *(_QWORD *)(a1 + 120) = sub_A7B3B0((__int64 *)(a1 + 120), v8, -1, a2, a3, v9, a4, a5);
  return v11;
}
