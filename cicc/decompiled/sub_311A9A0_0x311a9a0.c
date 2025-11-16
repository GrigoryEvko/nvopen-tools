// Function: sub_311A9A0
// Address: 0x311a9a0
//
__int64 __fastcall sub_311A9A0(__int64 a1, int a2)
{
  _QWORD *v2; // rcx
  _QWORD *v3; // rdx
  _QWORD *v4; // rax
  __int64 v5; // r8
  unsigned __int64 v6; // rcx
  _QWORD *v8; // rcx
  _QWORD *v9; // rdx
  _QWORD *v10; // rax
  __int64 v11; // rcx

  if ( a2 == 1 )
  {
    if ( *(_DWORD *)(a1 + 16) )
    {
      v8 = *(_QWORD **)(a1 + 8);
      v9 = &v8[9 * *(unsigned int *)(a1 + 24)];
      if ( v8 != v9 )
      {
        while ( 1 )
        {
          v10 = v8;
          if ( *v8 <= 0xFFFFFFFFFFFFFFFDLL )
            break;
          v8 += 9;
          if ( v9 == v8 )
            return 0;
        }
        if ( v9 != v8 )
        {
          v5 = 0;
          while ( 1 )
          {
            v11 = *((unsigned int *)v10 + 4);
            v10 += 9;
            v5 += v11;
            if ( v10 == v9 )
              return v5;
            while ( *v10 > 0xFFFFFFFFFFFFFFFDLL )
            {
              v10 += 9;
              if ( v9 == v10 )
                return v5;
            }
            if ( v10 == v9 )
              return v5;
          }
        }
      }
    }
    return 0;
  }
  if ( a2 == 2 )
  {
    if ( *(_DWORD *)(a1 + 16) )
    {
      v2 = *(_QWORD **)(a1 + 8);
      v3 = &v2[9 * *(unsigned int *)(a1 + 24)];
      if ( v2 != v3 )
      {
        while ( 1 )
        {
          v4 = v2;
          if ( *v2 <= 0xFFFFFFFFFFFFFFFDLL )
            break;
          v2 += 9;
          if ( v3 == v2 )
            return 0;
        }
        if ( v2 != v3 )
        {
          v5 = 0;
          do
          {
            v6 = *((unsigned int *)v4 + 4);
            if ( v6 > 1 )
              v5 += v6;
            v4 += 9;
            if ( v4 == v3 )
              break;
            while ( *v4 > 0xFFFFFFFFFFFFFFFDLL )
            {
              v4 += 9;
              if ( v3 == v4 )
                return v5;
            }
          }
          while ( v4 != v3 );
          return v5;
        }
      }
    }
    return 0;
  }
  if ( a2 )
    BUG();
  return *(unsigned int *)(a1 + 16);
}
