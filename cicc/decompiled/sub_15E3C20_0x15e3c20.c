// Function: sub_15E3C20
// Address: 0x15e3c20
//
__int64 __fastcall sub_15E3C20(_QWORD *a1)
{
  __int64 v1; // r13
  _QWORD *v2; // r13
  _QWORD *v3; // rbx
  unsigned __int64 *v4; // rcx
  unsigned __int64 v5; // rdx

  sub_15E0C30((__int64)a1);
  if ( a1[11] )
    sub_15E09A0((__int64)a1);
  sub_15E3BD0((__int64)a1);
  v1 = a1[13];
  if ( v1 )
  {
    sub_164D180(a1[13]);
    j_j___libc_free_0(v1, 40);
  }
  v2 = (_QWORD *)a1[10];
  while ( a1 + 9 != v2 )
  {
    v3 = v2;
    v2 = (_QWORD *)v2[1];
    sub_15E0220((__int64)(a1 + 9), (__int64)(v3 - 3));
    v4 = (unsigned __int64 *)v3[1];
    v5 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
    *v4 = v5 | *v4 & 7;
    *(_QWORD *)(v5 + 8) = v4;
    *v3 &= 7uLL;
    v3[1] = 0;
    sub_157EF40((__int64)(v3 - 3));
    j_j___libc_free_0(v3 - 3, 64);
  }
  sub_159D9E0((__int64)a1);
  return sub_164BE60(a1);
}
