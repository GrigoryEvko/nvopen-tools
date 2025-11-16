// Function: sub_29D88C0
// Address: 0x29d88c0
//
__int64 __fastcall sub_29D88C0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 result; // rax

  if ( a2 == a3 )
    return 0;
  v4 = sub_B3B7D0(a3);
  v5 = sub_B3B7D0(a2);
  result = sub_29D81B0(a1, v5, v4);
  if ( !(_DWORD)result )
  {
    result = sub_29D7F50(
               (__int64)a1,
               *(const void **)(a2 + 24),
               *(_QWORD *)(a2 + 32),
               *(const void **)(a3 + 24),
               *(_QWORD *)(a3 + 32));
    if ( !(_DWORD)result )
    {
      result = sub_29D7F50(
                 (__int64)a1,
                 *(const void **)(a2 + 56),
                 *(_QWORD *)(a2 + 64),
                 *(const void **)(a3 + 56),
                 *(_QWORD *)(a3 + 64));
      if ( !(_DWORD)result )
      {
        result = sub_29D7CF0((__int64)a1, *(unsigned __int8 *)(a2 + 96), *(unsigned __int8 *)(a3 + 96));
        if ( !(_DWORD)result )
        {
          result = sub_29D7CF0((__int64)a1, *(unsigned __int8 *)(a2 + 97), *(unsigned __int8 *)(a3 + 97));
          if ( !(_DWORD)result )
            return sub_29D7CF0((__int64)a1, *(unsigned int *)(a2 + 100), *(unsigned int *)(a3 + 100));
        }
      }
    }
  }
  return result;
}
