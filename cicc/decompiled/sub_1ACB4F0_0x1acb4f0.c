// Function: sub_1ACB4F0
// Address: 0x1acb4f0
//
__int64 __fastcall sub_1ACB4F0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 result; // rax

  if ( a2 == a3 )
    return 0;
  v4 = sub_15EAB70(a3);
  v5 = sub_15EAB70(a2);
  result = sub_1ACB220(a1, v5, v4);
  if ( !(_DWORD)result )
  {
    result = sub_1ACABE0(
               (__int64)a1,
               *(const void **)(a2 + 24),
               *(_QWORD *)(a2 + 32),
               *(const void **)(a3 + 24),
               *(_QWORD *)(a3 + 32));
    if ( !(_DWORD)result )
    {
      result = sub_1ACABE0(
                 (__int64)a1,
                 *(const void **)(a2 + 56),
                 *(_QWORD *)(a2 + 64),
                 *(const void **)(a3 + 56),
                 *(_QWORD *)(a3 + 64));
      if ( !(_DWORD)result )
      {
        result = sub_1ACA9E0((__int64)a1, *(unsigned __int8 *)(a2 + 96), *(unsigned __int8 *)(a3 + 96));
        if ( !(_DWORD)result )
        {
          result = sub_1ACA9E0((__int64)a1, *(unsigned __int8 *)(a2 + 97), *(unsigned __int8 *)(a3 + 97));
          if ( !(_DWORD)result )
            return sub_1ACA9E0((__int64)a1, *(unsigned int *)(a2 + 100), *(unsigned int *)(a3 + 100));
        }
      }
    }
  }
  return result;
}
