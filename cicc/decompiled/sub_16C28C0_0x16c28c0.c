// Function: sub_16C28C0
// Address: 0x16c28c0
//
_QWORD *__fastcall sub_16C28C0(_QWORD *a1, const void *a2, unsigned __int64 a3, __int64 a4)
{
  __int64 v5; // [rsp+0h] [rbp-30h] BYREF
  char v6; // [rsp+10h] [rbp-20h]

  sub_16C26E0((__int64)&v5, a2, a3, a4);
  if ( (v6 & 1) != 0 )
    *a1 = 0;
  else
    *a1 = v5;
  return a1;
}
