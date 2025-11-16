// Function: sub_88FC40
// Address: 0x88fc40
//
__int64 *__fastcall sub_88FC40(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 *result; // rax
  __int64 i; // rdx
  _QWORD *v5; // r8
  __int64 *v6; // rdx
  unsigned int v7; // esi
  unsigned __int64 v8; // rcx
  char v9; // si

  for ( result = a3; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  for ( i = *(_QWORD *)(a2 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v5 = (_QWORD *)result[36];
  if ( v5 )
  {
    v6 = **(__int64 ***)(i + 168);
    result = **(__int64 ***)(a1 + 168);
    if ( v6 )
    {
      if ( result )
      {
        while ( 2 )
        {
          v7 = *((_DWORD *)result + 9);
          do
          {
            v8 = *((unsigned int *)v6 + 9);
            if ( v7 != (_DWORD)v8 )
            {
              do
              {
                result = (__int64 *)*result;
                if ( !result )
                  return result;
              }
              while ( (unsigned int)v8 > *((_DWORD *)result + 9) );
            }
            if ( (result[4] & 4) != 0 && v5 && v5[7] <= v8 )
            {
              v9 = *((_BYTE *)v6 + 32) | 4;
              *((_BYTE *)v6 + 32) = v9;
              *((_BYTE *)v6 + 32) = v9 & 0xE7 | result[4] & 8 | 0x10;
              v6[6] = v5[6];
            }
            v6 = (__int64 *)*v6;
            if ( !v6 )
              return result;
            v7 = *((_DWORD *)result + 9);
          }
          while ( *((_DWORD *)v6 + 9) <= v7 );
          if ( v5 )
          {
            if ( (result[4] & 4) != 0 )
              v5 = (_QWORD *)*v5;
          }
          result = (__int64 *)*result;
          if ( result )
            continue;
          break;
        }
      }
    }
  }
  return result;
}
