// Function: sub_3512080
// Address: 0x3512080
//
__int64 __fastcall sub_3512080(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rax
  _QWORD *v4; // rdx
  __int64 *v6; // rbx
  __int64 *i; // r13
  __int64 v8; // rsi
  _QWORD *v9; // rax
  _QWORD *v10; // rdx

  if ( *(_BYTE *)(a2 + 28) )
  {
    v3 = *(_QWORD **)(a2 + 8);
    v4 = &v3[*(unsigned int *)(a2 + 20)];
    if ( v3 == v4 )
      goto LABEL_8;
    while ( a1 != *v3 )
    {
      if ( v4 == ++v3 )
        goto LABEL_8;
    }
    return 0;
  }
  if ( sub_C8CA60(a2, a1) )
    return 0;
LABEL_8:
  v6 = *(__int64 **)(a1 + 112);
  for ( i = &v6[*(unsigned int *)(a1 + 120)]; i != v6; ++v6 )
  {
    while ( 1 )
    {
      v8 = *v6;
      if ( !*(_BYTE *)(a2 + 28) )
        break;
      v9 = *(_QWORD **)(a2 + 8);
      v10 = &v9[*(unsigned int *)(a2 + 20)];
      if ( v9 == v10 )
        return 0;
      while ( v8 != *v9 )
      {
        if ( v10 == ++v9 )
          return 0;
      }
      if ( i == ++v6 )
        return 1;
    }
    if ( !sub_C8CA60(a2, v8) )
      return 0;
  }
  return 1;
}
