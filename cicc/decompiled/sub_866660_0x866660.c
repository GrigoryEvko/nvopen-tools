// Function: sub_866660
// Address: 0x866660
//
__int64 __fastcall sub_866660(__int64 a1)
{
  int v2; // edi
  __int64 result; // rax
  _QWORD *v4; // rcx
  __int64 *v5; // rax
  __int64 v6; // rdx

  v2 = *(_DWORD *)(a1 + 56);
  result = 0;
  if ( v2 )
  {
    v4 = qword_4F04C18;
    if ( qword_4F04C18 )
    {
      do
      {
        if ( *((_BYTE *)v4 + 42) )
          break;
        result = v4[2];
        if ( !result )
          return result;
        if ( *((_BYTE *)v4 + 40) )
        {
          v5 = *(__int64 **)(result + 8);
          if ( v5 )
          {
            while ( 1 )
            {
              if ( *((_DWORD *)v5 + 8) == 2 )
              {
                v6 = v5[7];
                if ( v6 )
                {
                  if ( *(_QWORD *)(v6 + 8)
                    && v2 == *(_DWORD *)(*(_QWORD *)(v6 + 16) + 36LL)
                    && *(_DWORD *)(v6 + 24) == *(_DWORD *)(a1 + 60) )
                  {
                    break;
                  }
                }
              }
              v5 = (__int64 *)*v5;
              if ( !v5 )
                goto LABEL_4;
            }
            result = *(_QWORD *)(v5[10] + 8);
            if ( result )
              return result;
          }
        }
LABEL_4:
        v4 = (_QWORD *)*v4;
      }
      while ( v4 );
      return 0;
    }
  }
  return result;
}
