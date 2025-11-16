// Function: sub_2304740
// Address: 0x2304740
//
_QWORD *__fastcall sub_2304740(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 v5; // r13
  _QWORD *v6; // rax
  __int64 *v7; // r13
  __int64 v9[8]; // [rsp+0h] [rbp-40h] BYREF

  sub_10497B0(v9, a2 + 8, a3, a4);
  v4 = v9[0];
  v5 = v9[1];
  v6 = (_QWORD *)sub_22077B0(0x20u);
  if ( v6 )
  {
    v6[1] = v4;
    v6[2] = v5;
    v6[3] = 0;
    *v6 = &unk_4A0AED0;
  }
  v7 = (__int64 *)v9[2];
  *a1 = v6;
  if ( v7 )
  {
    sub_FDC110(v7);
    j_j___libc_free_0((unsigned __int64)v7);
  }
  return a1;
}
