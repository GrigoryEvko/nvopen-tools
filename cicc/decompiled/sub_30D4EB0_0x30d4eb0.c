// Function: sub_30D4EB0
// Address: 0x30d4eb0
//
__int64 __fastcall sub_30D4EB0(__int64 a1, const void *a2, size_t a3)
{
  __int64 v4; // rax
  __int64 v6; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v7[0] = *(_QWORD *)(a1 + 72);
  v4 = sub_A747B0(v7, -1, a2, a3);
  if ( !v4 )
    v4 = sub_B49600(a1, a2, a3);
  v6 = v4;
  return sub_30D4E50(&v6);
}
