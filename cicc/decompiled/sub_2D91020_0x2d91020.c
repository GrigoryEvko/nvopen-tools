// Function: sub_2D91020
// Address: 0x2d91020
//
__int64 **__fastcall sub_2D91020(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  unsigned __int64 v4; // r12
  __int64 *v5; // rbx
  __int64 v6; // r14
  __int64 i; // r12

  v3 = qword_501CE50;
  v4 = *(_QWORD *)(qword_501CE50 + 144) - *(_QWORD *)(qword_501CE50 + 136);
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( v4 )
  {
    if ( v4 > 0x7FFFFFFFFFFFFFE0LL )
      sub_4261EA(a1, a2, a3);
    v5 = (__int64 *)sub_22077B0(v4);
  }
  else
  {
    v5 = 0;
  }
  *a1 = v5;
  a1[1] = v5;
  a1[2] = (__int64 *)((char *)v5 + v4);
  v6 = *(_QWORD *)(v3 + 144);
  for ( i = *(_QWORD *)(v3 + 136); v6 != i; v5 += 4 )
  {
    if ( v5 )
    {
      *v5 = (__int64)(v5 + 2);
      sub_2D8E2E0(v5, *(_BYTE **)i, *(_QWORD *)i + *(_QWORD *)(i + 8));
    }
    i += 32;
  }
  a1[1] = v5;
  return a1;
}
