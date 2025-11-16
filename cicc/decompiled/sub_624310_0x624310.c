// Function: sub_624310
// Address: 0x624310
//
__int64 __fastcall sub_624310(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 i; // rdx
  _QWORD *v4; // rdx
  __int64 v5; // rcx

  result = a1;
  if ( *(_BYTE *)(a1 + 140) != 12 )
  {
    result = *(_QWORD *)(a2 + 80);
    if ( !result )
    {
      result = sub_73EDA0(a1, 0);
      for ( i = result; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v4 = *(_QWORD **)(i + 168);
      if ( !dword_4D048B8 )
      {
        if ( v4[7] )
          v4[7] = 0;
      }
      while ( 1 )
      {
        v4 = (_QWORD *)*v4;
        if ( !v4 )
          break;
        while ( 1 )
        {
          v5 = v4[2];
          if ( !v5 )
            break;
          v4[1] = v5;
          v4 = (_QWORD *)*v4;
          if ( !v4 )
            return result;
        }
      }
    }
  }
  return result;
}
