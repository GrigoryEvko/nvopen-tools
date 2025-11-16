// Function: sub_1713E20
// Address: 0x1713e20
//
void __fastcall sub_1713E20(__int64 *a1, __int64 *a2, __int64 (__fastcall *a3)(__int64, __int64))
{
  __int64 v4; // rcx
  char *v5; // r15
  __int64 v6; // rbx

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_1713AE0(a1, a2, a3);
  }
  else
  {
    v4 = ((char *)a2 - (char *)a1) >> 4;
    v5 = (char *)&a1[v4];
    v6 = (8 * v4) >> 3;
    sub_1713E20(a1, v5);
    sub_1713E20(v5, a2);
    sub_1713CA0(
      (char *)a1,
      v5,
      (__int64)a2,
      v6,
      ((char *)a2 - v5) >> 3,
      (unsigned __int8 (__fastcall *)(_QWORD, _QWORD))a3);
  }
}
