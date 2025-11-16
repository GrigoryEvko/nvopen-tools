// Function: sub_30FD170
// Address: 0x30fd170
//
__int64 *__fastcall sub_30FD170(__int64 *a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // rbx

  v4 = sub_30CC5F0(a2, (__int64)a3);
  v5 = sub_22077B0(0x228u);
  v6 = v5;
  if ( v5 )
    sub_30FCF60(v5, a2, a3, v4, 1);
  *a1 = v6;
  return a1;
}
