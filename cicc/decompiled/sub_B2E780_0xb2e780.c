// Function: sub_B2E780
// Address: 0xb2e780
//
__int64 __fastcall sub_B2E780(_QWORD *a1)
{
  __int64 v1; // r13
  _QWORD *v2; // r13
  _QWORD *v3; // rbx
  unsigned __int64 *v4; // rcx
  unsigned __int64 v5; // rdx

  nullsub_59();
  sub_B2CA40((__int64)a1, 1);
  if ( a1[12] )
    sub_B2C790((__int64)a1);
  sub_B2E730((__int64)a1);
  v1 = a1[14];
  if ( v1 )
  {
    sub_BD84F0(a1[14]);
    j_j___libc_free_0(v1, 32);
  }
  v2 = (_QWORD *)a1[10];
  while ( a1 + 9 != v2 )
  {
    v3 = v2;
    v2 = (_QWORD *)v2[1];
    sub_B2B7E0((__int64)(a1 + 9), (__int64)(v3 - 3));
    v4 = (unsigned __int64 *)v3[1];
    v5 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
    *v4 = v5 | *v4 & 7;
    *(_QWORD *)(v5 + 8) = v4;
    *v3 &= 7uLL;
    v3[1] = 0;
    sub_AA5290((__int64)(v3 - 3));
    j_j___libc_free_0(v3 - 3, 80);
  }
  return sub_B2F9E0(a1);
}
