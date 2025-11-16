// Function: sub_3703E00
// Address: 0x3703e00
//
__int64 *__fastcall sub_3703E00(__int64 *a1, int *a2)
{
  int v2; // r13d
  __int64 v3; // rax
  __int64 v4; // rbx

  v2 = *a2;
  v3 = sub_22077B0(0x30u);
  v4 = v3;
  if ( v3 )
    sub_12547D0(v3, v2);
  *a1 = v4 | 1;
  return a1;
}
