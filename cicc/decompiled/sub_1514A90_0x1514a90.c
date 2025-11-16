// Function: sub_1514A90
// Address: 0x1514a90
//
__int64 __fastcall sub_1514A90(__int64 *a1, __int64 *a2)
{
  __int64 v4; // rcx
  unsigned __int64 v5; // rdx
  __int64 *v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rcx
  _QWORD *v9; // rdx
  __int64 result; // rax
  __int64 j; // rcx
  _QWORD *v12; // rdx
  __int64 v13; // rcx
  _QWORD *v14; // rdx
  __int64 i; // rcx
  _QWORD *v16; // rdx

  v4 = a1[3];
  v5 = a2[3];
  v6 = (__int64 *)(v4 + 8);
  if ( v4 + 8 < v5 )
  {
    do
    {
      v7 = *v6;
      v8 = *v6 + 512;
      do
      {
        v9 = *(_QWORD **)(v7 + 8);
        if ( v9 )
          *v9 = 0;
        v7 += 16;
      }
      while ( v8 != v7 );
      v5 = a2[3];
      ++v6;
    }
    while ( v5 > (unsigned __int64)v6 );
    v4 = a1[3];
  }
  result = *a1;
  if ( v5 == v4 )
  {
    for ( i = *a2; i != result; result += 16 )
    {
      v16 = *(_QWORD **)(result + 8);
      if ( v16 )
        *v16 = 0;
    }
  }
  else
  {
    for ( j = a1[2]; j != result; result += 16 )
    {
      v12 = *(_QWORD **)(result + 8);
      if ( v12 )
        *v12 = 0;
    }
    v13 = *a2;
    result = a2[1];
    if ( *a2 != result )
    {
      do
      {
        v14 = *(_QWORD **)(result + 8);
        if ( v14 )
          *v14 = 0;
        result += 16;
      }
      while ( v13 != result );
    }
  }
  return result;
}
