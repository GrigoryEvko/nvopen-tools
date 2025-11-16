// Function: sub_927750
// Address: 0x927750
//
__int64 __fastcall sub_927750(__int64 a1, __int64 a2, __int64 *a3)
{
  _QWORD *v3; // rax
  __int64 v4; // r12
  __int64 i; // rbx
  _QWORD *v6; // r14

  v3 = sub_7312D0(*(_QWORD *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 272LL));
  v3[2] = 0;
  v4 = (__int64)v3;
  for ( i = *(_QWORD *)(qword_4F04C50 + 40LL); i; i = *(_QWORD *)(i + 112) )
  {
    v6 = v3;
    v3 = sub_73E830(i);
    v6[2] = v3;
    v3[2] = 0;
  }
  sub_73DBF0(0x69u, *a3, v4);
  sub_926600(a1);
  return a1;
}
