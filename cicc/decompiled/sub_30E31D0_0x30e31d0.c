// Function: sub_30E31D0
// Address: 0x30e31d0
//
__int64 *__fastcall sub_30E31D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v6; // r12
  __int64 v7; // r14
  __int64 *v9; // rbx
  bool v10; // zf
  char v11; // r10
  __int64 *result; // rax
  __int64 v13; // [rsp+8h] [rbp-58h]
  __int64 v15; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v16[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = a1;
  v6 = a2;
  v7 = (a2 - 1) / 2;
  v13 = a3;
  if ( a2 <= a3 )
  {
    result = (__int64 *)(a1 + 8 * a2);
  }
  else
  {
    while ( 1 )
    {
      v9 = (__int64 *)(v5 + 8 * v7);
      v10 = *(_QWORD *)(a5 + 16) == 0;
      v15 = *v9;
      v16[0] = a4;
      if ( v10 )
        sub_4263D6(a1, a2, a3);
      a2 = (__int64)&v15;
      a1 = a5;
      v11 = (*(__int64 (__fastcall **)(__int64, __int64 *, _QWORD *))(a5 + 24))(a5, &v15, v16);
      result = (__int64 *)(v5 + 8 * v6);
      if ( !v11 )
        break;
      v6 = v7;
      *result = *v9;
      a3 = v7 - 1;
      if ( v13 >= v7 )
      {
        result = (__int64 *)(v5 + 8 * v7);
        break;
      }
      v7 = (v7 - 1) / 2;
    }
  }
  *result = a4;
  return result;
}
