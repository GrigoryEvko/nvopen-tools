// Function: sub_2305580
// Address: 0x2305580
//
void __fastcall sub_2305580(_QWORD *a1)
{
  __int64 v1; // rsi

  v1 = a1[1];
  *a1 = &unk_4A0AD40;
  if ( v1 )
    sub_10568E0((__int64)(a1 + 1), v1);
  j_j___libc_free_0((unsigned __int64)a1);
}
