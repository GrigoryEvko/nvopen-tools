// Function: sub_161FC80
// Address: 0x161fc80
//
bool __fastcall sub_161FC80(__int64 a1, __int64 *a2)
{
  __int64 *v2; // rax
  __int64 v3; // rdx
  __int64 *v4; // rcx
  __int64 v5; // rdx
  __int64 v7; // rsi

  v2 = *(__int64 **)(a1 + 16);
  if ( v2 == *(__int64 **)(a1 + 8) )
    v3 = *(unsigned int *)(a1 + 28);
  else
    v3 = *(unsigned int *)(a1 + 24);
  v4 = &v2[v3];
  if ( v2 != v4 )
  {
    while ( 1 )
    {
      v5 = *v2;
      if ( (unsigned __int64)*v2 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( ++v2 == v4 )
        return v2 != v4;
    }
    if ( v2 != v4 )
    {
      v7 = *a2;
LABEL_9:
      if ( v7 != v5 )
      {
        while ( ++v2 != v4 )
        {
          v5 = *v2;
          if ( (unsigned __int64)*v2 < 0xFFFFFFFFFFFFFFFELL )
          {
            if ( v2 != v4 )
              goto LABEL_9;
            return v2 != v4;
          }
        }
      }
    }
  }
  return v2 != v4;
}
