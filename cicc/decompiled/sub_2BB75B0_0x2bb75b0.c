// Function: sub_2BB75B0
// Address: 0x2bb75b0
//
void __fastcall sub_2BB75B0(__int64 *a1, __int64 *a2, __int64 (__fastcall *a3)(__int64, __int64, __int64), __int64 a4)
{
  __int64 v6; // rax
  char *v7; // rbx
  __int64 v8; // r9
  __int64 v9; // [rsp+8h] [rbp-38h]

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_2B82060(a1, a2, a3, a4);
  }
  else
  {
    v6 = ((char *)a2 - (char *)a1) >> 4;
    v7 = (char *)&a1[v6];
    v9 = v6 * 8;
    sub_2BB75B0(a1, &a1[v6], a3, a4);
    sub_2BB75B0(v7, a2, a3, a4);
    sub_2BB73F0(
      (char *)a1,
      v7,
      (__int64)a2,
      v9 >> 3,
      ((char *)a2 - v7) >> 3,
      v8,
      (unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD))a3,
      a4);
  }
}
