// Function: sub_1E6D460
// Address: 0x1e6d460
//
__int64 *__fastcall sub_1E6D460(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // r14
  __int64 *v8; // r13
  __int64 *result; // rax
  __int64 *v11; // [rsp+8h] [rbp-38h]

  v5 = a2;
  v6 = (a2 - 1) / 2;
  v11 = (__int64 *)a5;
  if ( a2 > a3 )
  {
    while ( 1 )
    {
      v8 = (__int64 *)(a1 + 8 * v6);
      LODWORD(a5) = sub_1E6D280(v11, *v8, a4, a4, a5);
      result = (__int64 *)(a1 + 8 * v5);
      if ( !(_BYTE)a5 )
        break;
      v5 = v6;
      *result = *v8;
      if ( a3 >= v6 )
      {
        result = (__int64 *)(a1 + 8 * v6);
        break;
      }
      v6 = (v6 - 1) / 2;
    }
  }
  else
  {
    result = (__int64 *)(a1 + 8 * a2);
  }
  *result = a4;
  return result;
}
