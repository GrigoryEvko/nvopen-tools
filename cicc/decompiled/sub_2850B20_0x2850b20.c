// Function: sub_2850B20
// Address: 0x2850b20
//
__int64 __fastcall sub_2850B20(__int64 a1, int a2)
{
  int v2; // r15d
  __int16 v4; // ax
  __int64 *v6; // r12
  __int64 *v7; // r13
  int v8; // ebx
  __int64 v9; // rdi
  int v10; // eax

  v2 = 0;
  while ( 2 )
  {
    while ( 2 )
    {
      v4 = *(_WORD *)(a1 + 24);
      if ( v4 == 15 )
      {
LABEL_7:
        a2 = 1;
        return (unsigned int)(v2 + a2);
      }
      while ( 1 )
      {
        if ( !v4 )
          goto LABEL_7;
        if ( !a2 )
          return (unsigned int)(v2 + a2);
        if ( v4 != 8 )
          break;
        --a2;
        a1 = **(_QWORD **)(a1 + 32);
        v4 = *(_WORD *)(a1 + 24);
        if ( v4 == 15 )
          goto LABEL_7;
      }
      if ( (unsigned __int16)(v4 - 2) <= 2u )
      {
        a1 = *(_QWORD *)(a1 + 32);
        --a2;
        continue;
      }
      break;
    }
    if ( (unsigned __int16)(v4 - 9) > 4u && (unsigned __int16)(v4 - 5) > 1u )
    {
      if ( v4 == 7 )
      {
        v10 = sub_2850B20(*(_QWORD *)(a1 + 32));
        a1 = *(_QWORD *)(a1 + 40);
        --a2;
        v2 += v10;
        continue;
      }
      goto LABEL_19;
    }
    break;
  }
  v6 = *(__int64 **)(a1 + 32);
  v7 = &v6[*(_QWORD *)(a1 + 40)];
  if ( v7 == v6 )
  {
LABEL_19:
    a2 = 0;
    return (unsigned int)(v2 + a2);
  }
  v8 = 0;
  do
  {
    v9 = *v6++;
    a2 = v8 + sub_2850B20(v9);
    v8 = a2;
  }
  while ( v7 != v6 );
  return (unsigned int)(v2 + a2);
}
