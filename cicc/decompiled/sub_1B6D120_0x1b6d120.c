// Function: sub_1B6D120
// Address: 0x1b6d120
//
__int64 __fastcall sub_1B6D120(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  __int64 v5; // r12
  __int64 *v6; // r13
  __int64 *v7; // rbx
  __int64 *v8; // r14
  __int64 *v9; // rsi
  __int64 *v10; // r12
  unsigned __int64 v11; // [rsp+8h] [rbp-48h]
  char v12[49]; // [rsp+1Fh] [rbp-31h] BYREF

  result = qword_4FB7768 - qword_4FB7760;
  v11 = qword_4FB7768 - qword_4FB7760;
  if ( qword_4FB7768 != qword_4FB7760 )
  {
    if ( v11 > 0x7FFFFFFFFFFFFFE0LL )
      sub_4261EA(a1, a2, a3);
    result = sub_22077B0(v11);
    v4 = qword_4FB7768;
    v5 = qword_4FB7760;
    v6 = (__int64 *)result;
    if ( qword_4FB7768 != qword_4FB7760 )
    {
      v7 = (__int64 *)result;
      do
      {
        if ( v7 )
        {
          *v7 = (__int64)(v7 + 2);
          sub_1B67C80(v7, *(_BYTE **)v5, *(_QWORD *)v5 + *(_QWORD *)(v5 + 8));
        }
        v5 += 32;
        v7 += 4;
      }
      while ( v4 != v5 );
      if ( v7 == v6 )
        return j_j___libc_free_0(v6, v11);
      v8 = v6;
      do
      {
        v9 = v8;
        v8 += 4;
        sub_1B6CF20((__int64)v12, v9, a1);
      }
      while ( v7 != v8 );
      v10 = v6;
      do
      {
        result = (__int64)(v10 + 2);
        if ( (__int64 *)*v10 != v10 + 2 )
          result = j_j___libc_free_0(*v10, v10[2] + 1);
        v10 += 4;
      }
      while ( v7 != v10 );
    }
    if ( !v6 )
      return result;
    return j_j___libc_free_0(v6, v11);
  }
  return result;
}
