// Function: sub_1056970
// Address: 0x1056970
//
__int64 __fastcall sub_1056970(_QWORD *a1)
{
  __int64 v1; // rsi

  v1 = a1[23];
  *a1 = &unk_49E5D58;
  if ( v1 )
    sub_10568E0((__int64)(a1 + 23), v1);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  return j_j___libc_free_0(a1, 192);
}
