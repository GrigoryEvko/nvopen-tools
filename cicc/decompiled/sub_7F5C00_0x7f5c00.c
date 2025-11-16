// Function: sub_7F5C00
// Address: 0x7f5c00
//
__int64 __fastcall sub_7F5C00(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdi
  __int64 result; // rax
  __int64 i; // rbx
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rbx

  v1 = a1;
  v2 = *(_QWORD *)(a1 + 128);
  if ( v2 )
  {
LABEL_7:
    if ( *(_BYTE *)(v1 + 173) == 10 )
    {
      result = sub_8D3410(v2);
      if ( (_DWORD)result )
      {
        v5 = sub_8D40F0(*(_QWORD *)(v1 + 128));
        result = (__int64)&dword_4F077C4;
        if ( dword_4F077C4 == 2 )
        {
          result = (__int64)&dword_4F06968;
          if ( !dword_4F06968 )
          {
            result = sub_7E1F90(v5);
            if ( !(_DWORD)result )
            {
              result = sub_7E6740(v5);
              if ( (_DWORD)result )
              {
                v6 = sub_8D4050(*(_QWORD *)(v1 + 128));
                result = sub_8D3410(v6);
                v7 = *(_QWORD *)(v1 + 176);
                if ( (_DWORD)result )
                {
                  while ( v7 )
                  {
                    result = sub_7F5C00(v7);
                    v7 = *(_QWORD *)(v7 + 120);
                  }
                }
                else
                {
                  while ( v7 )
                  {
                    if ( *(_BYTE *)(v7 + 173) == 11 )
                      result = sub_7F5C00(*(_QWORD *)(v7 + 176));
                    else
                      *(_BYTE *)(v7 + 168) |= 8u;
                    v7 = *(_QWORD *)(v7 + 120);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        for ( i = *(_QWORD *)(v1 + 176); i; i = *(_QWORD *)(i + 120) )
          result = sub_7F5C00(i);
      }
    }
    else
    {
      result = (__int64)&dword_4F077C4;
      if ( dword_4F077C4 == 2 )
      {
        result = dword_4F06968;
        if ( !dword_4F06968 )
        {
          result = sub_7E1F90(v2);
          if ( !(_DWORD)result )
          {
            result = sub_7E6740(*(_QWORD *)(v1 + 128));
            if ( (_DWORD)result )
              *(_BYTE *)(v1 + 168) |= 8u;
          }
        }
      }
    }
  }
  else
  {
    while ( 1 )
    {
      result = *(unsigned __int8 *)(v1 + 173);
      if ( (_BYTE)result != 11 )
        break;
      v1 = *(_QWORD *)(v1 + 176);
      v2 = *(_QWORD *)(v1 + 128);
      if ( v2 )
        goto LABEL_7;
    }
    if ( (_BYTE)result != 13 )
      sub_721090();
  }
  return result;
}
