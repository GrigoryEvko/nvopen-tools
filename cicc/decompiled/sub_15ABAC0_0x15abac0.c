// Function: sub_15ABAC0
// Address: 0x15abac0
//
__int64 __fastcall sub_15ABAC0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 *v4; // rbx
  char v5; // dl

  result = sub_15AB540(a1, a2);
  if ( (_BYTE)result )
  {
    sub_15AB790(a1, *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))));
    sub_15AB8A0(a1, *(_QWORD *)(a2 + 8 * (5LL - *(unsigned int *)(a2 + 8))));
    sub_15ABBA0(a1, *(_QWORD *)(a2 + 8 * (4LL - *(unsigned int *)(a2 + 8))));
    result = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)result > 9 )
    {
      v3 = *(_QWORD *)(a2 + 8 * (9 - result));
      if ( v3 )
      {
        result = 8LL * *(unsigned int *)(v3 + 8);
        v4 = (__int64 *)(v3 - result);
        if ( v3 - result != v3 )
        {
          do
          {
            while ( 1 )
            {
              result = *v4;
              v5 = *(_BYTE *)*v4;
              if ( v5 != 22 && v5 != 23 )
                break;
              ++v4;
              result = sub_15ABBA0(a1, *(_QWORD *)(result + 8 * (1LL - *(unsigned int *)(result + 8))));
              if ( (__int64 *)v3 == v4 )
                return result;
            }
            ++v4;
          }
          while ( (__int64 *)v3 != v4 );
        }
      }
    }
  }
  return result;
}
