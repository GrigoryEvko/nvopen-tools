// Function: sub_C80FA0
// Address: 0xc80fa0
//
unsigned __int8 *__fastcall sub_C80FA0(unsigned __int8 *a1, unsigned __int64 a2, unsigned int a3)
{
  unsigned __int64 v3; // rdx

  sub_C80E20(a1, a2, a3);
  if ( v3 <= a2 )
    a2 = v3;
  return &a1[a2];
}
