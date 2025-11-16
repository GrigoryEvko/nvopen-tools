// Function: sub_FE8080
// Address: 0xfe8080
//
__int64 *__fastcall sub_FE8080(__int64 *a1, __int64 a2, const char *a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // r14

  v6 = sub_BC1CD0(a4, &unk_4F8E5A8, (__int64)a3);
  v7 = sub_BC1CD0(a4, &unk_4F875F0, (__int64)a3);
  sub_FDC0F0(a1);
  sub_FE7D70(a1, a3, v6 + 8, v7 + 8);
  return a1;
}
