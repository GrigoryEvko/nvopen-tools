// Function: sub_F90050
// Address: 0xf90050
//
bool __fastcall sub_F90050(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 i; // rbx
  __int64 v6; // rsi
  _QWORD *v7; // rax
  _QWORD *v8; // rdx

  for ( i = a1; a2 != i; i = *(_QWORD *)(i + 8) )
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(i + 24);
      if ( *(_BYTE *)(a3 + 28) )
        break;
      if ( sub_C8CA60(a3, v6) )
      {
        i = *(_QWORD *)(i + 8);
        if ( a2 != i )
          continue;
      }
      return a2 == i;
    }
    v7 = *(_QWORD **)(a3 + 8);
    v8 = &v7[*(unsigned int *)(a3 + 20)];
    if ( v7 == v8 )
      break;
    while ( v6 != *v7 )
    {
      if ( v8 == ++v7 )
        return a2 == i;
    }
  }
  return a2 == i;
}
