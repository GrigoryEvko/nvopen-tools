// Function: sub_20C3470
// Address: 0x20c3470
//
__int64 __fastcall sub_20C3470(__int64 a1, unsigned int *a2)
{
  __int64 result; // rax
  __int64 v3; // r8
  unsigned int v4; // ecx
  __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 v7; // r9
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 i; // rsi

  result = *(_QWORD *)(a1 + 16);
  v3 = a1 + 8;
  if ( !result )
    return v3;
  v4 = *a2;
  while ( 1 )
  {
    while ( v4 > *(_DWORD *)(result + 32) )
    {
      result = *(_QWORD *)(result + 24);
      if ( !result )
        return v3;
    }
    v5 = *(_QWORD *)(result + 16);
    if ( v4 >= *(_DWORD *)(result + 32) )
      break;
    v3 = result;
    result = *(_QWORD *)(result + 16);
    if ( !v5 )
      return v3;
  }
  v6 = *(_QWORD *)(result + 24);
  if ( v6 )
  {
    do
    {
      while ( 1 )
      {
        v7 = *(_QWORD *)(v6 + 16);
        v8 = *(_QWORD *)(v6 + 24);
        if ( v4 < *(_DWORD *)(v6 + 32) )
          break;
        v6 = *(_QWORD *)(v6 + 24);
        if ( !v8 )
          goto LABEL_14;
      }
      v6 = *(_QWORD *)(v6 + 16);
    }
    while ( v7 );
  }
LABEL_14:
  while ( v5 )
  {
    v9 = *(_QWORD *)(v5 + 16);
    for ( i = *(_QWORD *)(v5 + 24); v4 > *(_DWORD *)(v5 + 32); i = *(_QWORD *)(i + 24) )
    {
      v5 = i;
      if ( !i )
        return result;
      v9 = *(_QWORD *)(i + 16);
    }
    result = v5;
    v5 = v9;
  }
  return result;
}
