// Function: sub_6DF600
// Address: 0x6df600
//
__int64 __fastcall sub_6DF600(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // r8
  __int64 v7; // rdx
  __int64 v8; // rcx

  result = *(unsigned int *)(a2 + 108);
  if ( (_DWORD)result )
  {
    result = *(_QWORD *)(a1 + 24);
    if ( result )
    {
      v3 = qword_4F06BC0;
      if ( result == qword_4F06BC0 )
      {
        v4 = *(_QWORD *)(a2 + 112);
        if ( v4 )
        {
          result = *(_QWORD *)(a1 + 32);
          if ( v4 != result )
          {
            v5 = *(_QWORD *)(qword_4F06BC0 + 24LL);
            if ( a1 == v5 )
            {
              *(_QWORD *)(qword_4F06BC0 + 24LL) = result;
            }
            else
            {
              do
              {
                v6 = v5;
                v5 = *(_QWORD *)(v5 + 32);
              }
              while ( a1 != v5 );
              *(_QWORD *)(v6 + 32) = result;
              result = *(_QWORD *)(v3 + 24);
            }
            v7 = *(_QWORD *)(a2 + 112);
            if ( result == v7 )
            {
              *(_QWORD *)(v3 + 24) = a1;
            }
            else
            {
              do
              {
                v8 = result;
                result = *(_QWORD *)(result + 32);
              }
              while ( v7 != result );
              *(_QWORD *)(v8 + 32) = a1;
            }
            result = *(_QWORD *)(a2 + 112);
            *(_QWORD *)(a1 + 32) = result;
          }
        }
        *(_QWORD *)(a2 + 112) = a1;
      }
    }
  }
  return result;
}
