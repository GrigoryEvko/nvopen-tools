// Function: sub_23AED50
// Address: 0x23aed50
//
bool __fastcall sub_23AED50(__int64 a1)
{
  __int64 *v1; // rax
  __int64 *v2; // rdx

  v1 = *(__int64 **)(a1 + 8);
  v2 = &v1[5 * *(unsigned int *)(a1 + 24)];
  if ( *(_DWORD *)(a1 + 16) )
  {
    if ( v1 != v2 )
    {
      while ( *v1 > 0x7FFFFFFFFFFFFFFDLL )
      {
        v1 += 5;
        if ( v2 == v1 )
          return v2 != v1;
      }
      if ( v2 != v1 )
      {
LABEL_9:
        if ( v1[4] )
        {
          while ( 1 )
          {
            v1 += 5;
            if ( v2 == v1 )
              break;
            if ( *v1 <= 0x7FFFFFFFFFFFFFFDLL )
            {
              if ( v2 != v1 )
                goto LABEL_9;
              return v2 != v1;
            }
          }
        }
      }
    }
  }
  else
  {
    v1 += 5 * *(unsigned int *)(a1 + 24);
  }
  return v2 != v1;
}
