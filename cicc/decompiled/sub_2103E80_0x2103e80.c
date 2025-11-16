// Function: sub_2103E80
// Address: 0x2103e80
//
__int64 __fastcall sub_2103E80(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r11
  __int64 v6; // r9
  unsigned __int16 *v7; // rdx
  unsigned int v8; // esi
  int v9; // edx

  result = *a1;
  v4 = *(unsigned int *)(*a1 + 44LL);
  if ( (_DWORD)v4 )
  {
    v6 = 0;
    while ( 1 )
    {
      if ( !result )
        BUG();
      v7 = (unsigned __int16 *)(*(_QWORD *)(result + 48) + 4 * v6);
      result = *v7;
      v8 = v7[1];
      if ( (_WORD)result )
      {
        while ( 1 )
        {
          v9 = *(_DWORD *)(a2 + 4 * ((unsigned __int64)(unsigned __int16)result >> 5));
          if ( _bittest(&v9, result) )
          {
            result = v8;
            if ( !(_WORD)v8 )
              break;
          }
          else
          {
            *(_QWORD *)(8LL * ((unsigned int)v6 >> 6) + a1[1]) &= ~(1LL << v6);
            result = v8;
            if ( !(_WORD)v8 )
              break;
          }
          v8 = 0;
        }
      }
      if ( ++v6 == v4 )
        break;
      result = *a1;
    }
  }
  return result;
}
