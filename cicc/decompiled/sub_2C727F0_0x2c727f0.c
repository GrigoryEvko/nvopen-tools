// Function: sub_2C727F0
// Address: 0x2c727f0
//
char __fastcall sub_2C727F0(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  __int16 v3; // cx
  char result; // al
  __int64 v5; // rcx
  _QWORD *v6; // r14
  unsigned int v7; // ebx
  __int64 v8; // r13

  while ( 1 )
  {
    v3 = *(_WORD *)(a1 + 24);
    result = v3 == 14 || (unsigned __int16)(v3 - 2) <= 2u;
    if ( !result )
      break;
    a1 = *(_QWORD *)(a1 + 32);
  }
  if ( (unsigned __int16)(v3 - 5) <= 1u || (unsigned __int16)(v3 - 8) <= 5u )
  {
    if ( v3 == 8 )
    {
      if ( a2 == *(_QWORD *)(a1 + 48) )
      {
        result = a3;
        if ( !a3 )
          return (*(_BYTE *)(a1 + 28) & 5) != 0;
      }
    }
    else if ( v3 == 5 )
    {
      v5 = *(_QWORD *)(a1 + 40);
      if ( (_DWORD)v5 )
      {
        v6 = *(_QWORD **)(a1 + 32);
        v7 = a3;
        v8 = (__int64)&v6[(unsigned int)(v5 - 1) + 1];
        do
        {
          result = sub_2C727F0(*v6, a2, v7);
          if ( result )
            break;
          ++v6;
        }
        while ( (_QWORD *)v8 != v6 );
      }
    }
  }
  return result;
}
