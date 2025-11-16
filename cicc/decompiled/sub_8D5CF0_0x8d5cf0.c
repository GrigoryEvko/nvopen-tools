// Function: sub_8D5CF0
// Address: 0x8d5cf0
//
__int64 **__fastcall sub_8D5CF0(__int64 a1, __int64 a2)
{
  __int64 **result; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx

  for ( result = **(__int64 ****)(a1 + 168); result; result = (__int64 **)*result )
  {
    if ( ((_BYTE)result[12] & 1) != 0 )
    {
      v3 = (__int64)result[5];
      if ( v3 == a2 )
        break;
      if ( v3 )
      {
        if ( a2 )
        {
          if ( dword_4F07588 )
          {
            v4 = *(_QWORD *)(v3 + 32);
            if ( *(_QWORD *)(a2 + 32) == v4 )
            {
              if ( v4 )
                break;
            }
          }
        }
      }
    }
  }
  return result;
}
