// Function: sub_1C3E900
// Address: 0x1c3e900
//
int __fastcall sub_1C3E900(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  _QWORD *v7; // r12
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9

  if ( !qword_4FBA5B0 )
    sub_16C1EA0((__int64)&qword_4FBA5B0, (__int64 (*)(void))sub_1C3E6D0, (__int64)sub_1C3E470, a4, a5, a6);
  v6 = sub_16D40F0(qword_4FBA5B0);
  v7 = v6;
  if ( v6 )
  {
    if ( (_QWORD *)*v6 != v6 + 2 )
      j_j___libc_free_0(*v6, v6[2] + 1LL);
    j_j___libc_free_0(v7, 32);
    if ( !qword_4FBA5B0 )
      sub_16C1EA0((__int64)&qword_4FBA5B0, (__int64 (*)(void))sub_1C3E6D0, (__int64)sub_1C3E470, v8, v9, v10);
    LODWORD(v6) = sub_16D40E0(qword_4FBA5B0, 0);
  }
  return (int)v6;
}
