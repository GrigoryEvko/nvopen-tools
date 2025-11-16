// Function: sub_37B9AB0
// Address: 0x37b9ab0
//
__int64 __fastcall sub_37B9AB0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax

  v2 = *(_DWORD *)(a1 + 56);
  if ( v2 != *(_DWORD *)(a2 + 56)
    || *(_QWORD *)(a1 + 40) != *(_QWORD *)(a2 + 40)
    || *(_BYTE *)(a1 + 48) != *(_BYTE *)(a2 + 48)
    || *(_BYTE *)(a1 + 49) != *(_BYTE *)(a2 + 49) )
  {
    return 0;
  }
  switch ( v2 )
  {
    case 1:
      v4 = *(unsigned int *)(a2 + 32);
      if ( v4 != *(_DWORD *)(a1 + 32) )
        return 0;
      v5 = a1 + 4 * v4;
      if ( a1 == v5 )
        return 1;
      while ( *(_DWORD *)a1 == *(_DWORD *)a2 )
      {
        a1 += 4;
        a2 += 4;
        if ( v5 == a1 )
          return 1;
      }
      return 0;
    case 3:
      result = 0;
      if ( *(_DWORD *)(a1 + 36) == *(_DWORD *)(a2 + 36) )
        return 1;
      break;
    case 2:
      result = 0;
      if ( *(_DWORD *)(a1 + 36) == *(_DWORD *)(a2 + 36) )
      {
        v6 = *(unsigned int *)(a2 + 32);
        if ( v6 == *(_DWORD *)(a1 + 32) )
        {
          v7 = a1 + 4 * v6;
          if ( a1 == v7 )
            return 1;
          while ( *(_DWORD *)a1 == *(_DWORD *)a2 )
          {
            a1 += 4;
            a2 += 4;
            if ( v7 == a1 )
              return 1;
          }
          return 0;
        }
        return 0;
      }
      break;
    default:
      return 1;
  }
  return result;
}
