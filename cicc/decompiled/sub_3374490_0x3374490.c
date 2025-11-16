// Function: sub_3374490
// Address: 0x3374490
//
__int64 __fastcall sub_3374490(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rdi
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 result; // rax
  __int64 i; // rdx

  v4 = *(_QWORD **)(a1 + 896);
  v6 = v4[4];
  v7 = v4[5];
  if ( v6 != v7 )
  {
    do
    {
      while ( *(_QWORD *)(v6 + 40) != a2 )
      {
        v6 += 104;
        if ( v7 == v6 )
          goto LABEL_6;
      }
      *(_QWORD *)(v6 + 40) = a3;
      v6 += 104;
    }
    while ( v7 != v6 );
LABEL_6:
    v4 = *(_QWORD **)(a1 + 896);
  }
  result = v4[7];
  for ( i = v4[8]; i != result; result += 192 )
  {
    while ( *(_QWORD *)(result + 48) != a2 )
    {
      result += 192;
      if ( i == result )
        return result;
    }
    *(_QWORD *)(result + 48) = a3;
  }
  return result;
}
