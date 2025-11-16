// Function: sub_25B0CC0
// Address: 0x25b0cc0
//
__int64 __fastcall sub_25B0CC0(_QWORD *a1, __int64 *a2)
{
  __int64 v3; // rax
  _QWORD *v4; // rdx
  _QWORD *v5; // r12
  __int64 v6; // rbx
  int *v7; // rdi
  int *v8; // rax
  __int64 v9; // rax
  __int64 v11; // [rsp+8h] [rbp-38h]

  v3 = sub_25B0BB0((__int64)a1, a2);
  v5 = v4;
  v6 = v3;
  v11 = a1[5];
  if ( v3 == a1[3] && v4 == a1 + 1 )
  {
    sub_25AE270(a1[2]);
    a1[2] = 0;
    a1[3] = v5;
    a1[4] = v5;
    a1[5] = 0;
  }
  else if ( (_QWORD *)v3 == v4 )
  {
    return 0;
  }
  else
  {
    do
    {
      v7 = (int *)v6;
      v6 = sub_220EF30(v6);
      v8 = sub_220F330(v7, a1 + 1);
      j_j___libc_free_0((unsigned __int64)v8);
      v9 = a1[5] - 1LL;
      a1[5] = v9;
    }
    while ( v5 != (_QWORD *)v6 );
    v11 -= v9;
  }
  return v11;
}
