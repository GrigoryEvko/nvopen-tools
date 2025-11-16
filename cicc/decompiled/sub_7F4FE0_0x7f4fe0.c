// Function: sub_7F4FE0
// Address: 0x7f4fe0
//
__int64 __fastcall sub_7F4FE0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 **v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rcx

  result = *(unsigned __int8 *)(a1 + 24);
  if ( (_BYTE)result == 3 )
  {
    v7 = *(_QWORD *)(a1 + 56);
    if ( *(char *)(v7 + 169) < 0 )
    {
      result = *(_QWORD *)(a2 + 152);
      v8 = *(_QWORD *)(a2 + 160);
      if ( result )
      {
        if ( v8 )
        {
          while ( v7 != result )
          {
            if ( (*(_BYTE *)(result + 172) & 1) != 0 )
              result = *(_QWORD *)(result + 112);
            v8 = *(_QWORD *)(v8 + 112);
            result = *(_QWORD *)(result + 112);
            if ( !result || !v8 )
              return result;
          }
          *(_QWORD *)(a1 + 56) = v8;
        }
      }
    }
  }
  else if ( (_BYTE)result == 6 )
  {
    result = *(_QWORD *)(a1 + 64);
    v5 = *(__int64 ***)result;
    if ( *(_QWORD *)result )
    {
      do
      {
        v6 = *(_QWORD *)(a2 + 160);
        result = *(_QWORD *)(a2 + 152);
        if ( v6 && result )
        {
          while ( ((_BYTE)v5[4] & 3) != 0 || v5[1] != (__int64 *)result )
          {
            if ( (*(_BYTE *)(result + 172) & 1) != 0 )
              result = *(_QWORD *)(result + 112);
            v6 = *(_QWORD *)(v6 + 112);
            result = *(_QWORD *)(result + 112);
            if ( !result || !v6 )
              goto LABEL_15;
          }
          v5[1] = (__int64 *)v6;
        }
LABEL_15:
        v5 = (__int64 **)*v5;
      }
      while ( v5 );
    }
  }
  return result;
}
