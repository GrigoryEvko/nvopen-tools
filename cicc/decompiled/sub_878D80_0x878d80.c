// Function: sub_878D80
// Address: 0x878d80
//
__int64 *__fastcall sub_878D80(unsigned int a1, unsigned __int64 a2)
{
  __int64 v3; // rsi
  __int64 *result; // rax
  unsigned int v5; // edx
  __int64 *v6; // rcx
  unsigned __int64 v7; // r9

  v3 = qword_4F04C68[0] + 776LL * unk_4F04C48;
  result = *(__int64 **)(v3 + 640);
  if ( !result )
    return result;
  v5 = *((_DWORD *)result + 4);
  if ( v5 >= a1 )
  {
    for ( ; v5 > a1; v5 = *((_DWORD *)result + 4) )
    {
      result = (__int64 *)*result;
      if ( !result )
        return result;
    }
LABEL_6:
    v6 = result;
    if ( a1 == v5 )
    {
      v7 = result[3];
      if ( v7 != a2 )
      {
        if ( v7 < a2 )
        {
          do
          {
            v6 = (__int64 *)v6[1];
            if ( !v6 )
              break;
            v5 = *((_DWORD *)v6 + 4);
            if ( v6[3] >= a2 )
              goto LABEL_26;
          }
          while ( a1 == v5 );
        }
        else
        {
          while ( a2 < v7 )
          {
            if ( a1 != v5 )
              return 0;
            v6 = (__int64 *)*v6;
            if ( !v6 )
              return 0;
            v7 = v6[3];
            v5 = *((_DWORD *)v6 + 4);
          }
LABEL_26:
          if ( a1 == v5 )
          {
            result = v6;
            if ( v6[3] == a2 )
              goto LABEL_8;
          }
        }
        return 0;
      }
    }
    else
    {
      result = 0;
    }
LABEL_8:
    *(_QWORD *)(v3 + 640) = v6;
    return result;
  }
  while ( 1 )
  {
    result = (__int64 *)result[1];
    if ( !result )
      return result;
    v5 = *((_DWORD *)result + 4);
    if ( v5 >= a1 )
      goto LABEL_6;
  }
}
