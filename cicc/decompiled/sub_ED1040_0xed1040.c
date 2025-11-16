// Function: sub_ED1040
// Address: 0xed1040
//
__int64 *__fastcall sub_ED1040(__int64 *a1, __int64 a2, int a3)
{
  __int64 v5[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v6[6]; // [rsp+10h] [rbp-30h] BYREF

  v5[0] = (__int64)v6;
  sub_ED0450(v5, byte_3F871B3, (__int64)byte_3F871B3);
  sub_ED0C50(a1, a3, (__int64)v5);
  if ( (_QWORD *)v5[0] != v6 )
    j_j___libc_free_0(v5[0], v6[0] + 1LL);
  return a1;
}
