// Function: sub_F18830
// Address: 0xf18830
//
void __fastcall sub_F18830(__int64 *a1, __int64 *a2, __int64 (__fastcall *a3)(__int64, __int64))
{
  __int64 v4; // rcx
  char *v5; // r15
  __int64 v6; // rbx

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_F181D0(a1, a2, a3);
  }
  else
  {
    v4 = ((char *)a2 - (char *)a1) >> 4;
    v5 = (char *)&a1[v4];
    v6 = (8 * v4) >> 3;
    sub_F18830(a1, v5);
    sub_F18830(v5, a2);
    sub_F186B0(
      (char *)a1,
      v5,
      (__int64)a2,
      v6,
      ((char *)a2 - v5) >> 3,
      (unsigned __int8 (__fastcall *)(_QWORD, _QWORD))a3);
  }
}
