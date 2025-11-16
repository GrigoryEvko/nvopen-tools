// Function: sub_225FFC0
// Address: 0x225ffc0
//
_QWORD *__fastcall sub_225FFC0(_QWORD *a1, __int64 **a2, __int64 a3)
{
  __int64 *v3; // rax
  __int64 v4; // rdx

  v3 = a2[1];
  if ( !v3[2] )
    sub_4263D6(a1, a2, a3);
  ((void (__fastcall *)(__int64 *))v3[3])(a2[1]);
  v4 = **a2;
  **a2 = 0;
  *a1 = v4;
  return a1;
}
