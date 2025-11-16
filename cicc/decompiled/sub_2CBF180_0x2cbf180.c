// Function: sub_2CBF180
// Address: 0x2cbf180
//
__int64 __fastcall sub_2CBF180(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 result; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx

  v8 = *(__int64 **)(a1 - 8);
  v9 = 4LL * *(unsigned int *)(a1 + 72);
  if ( a2 == v8[v9] )
  {
    v10 = *v8;
    result = 0;
    if ( *(_BYTE *)v10 <= 0x1Cu )
      return result;
  }
  else
  {
    if ( a2 != v8[v9 + 1] )
      BUG();
    v10 = v8[4];
    result = 0;
    if ( *(_BYTE *)v10 <= 0x1Cu )
      return result;
  }
  result = *(_QWORD *)(v10 + 16);
  if ( result )
  {
    while ( 1 )
    {
      v12 = *(_QWORD *)(result + 24);
      if ( *(_BYTE *)v12 == 84 )
      {
        v13 = *(_QWORD *)(v12 + 40);
        if ( a4 == v13 )
          break;
        if ( a3 == v13 )
        {
          v12 = *(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL);
          if ( *(_BYTE *)v12 == 84 && a4 == *(_QWORD *)(v12 + 40) )
            break;
        }
      }
      result = *(_QWORD *)(result + 8);
      if ( !result )
        return result;
    }
    return v12;
  }
  return result;
}
