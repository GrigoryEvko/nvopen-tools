// Function: sub_1FD3C80
// Address: 0x1fd3c80
//
unsigned __int64 __fastcall sub_1FD3C80(_QWORD *a1, _QWORD *a2)
{
  __int64 v3; // rcx
  _QWORD *v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  unsigned __int64 result; // rax
  __int64 v8; // rsi
  __int64 v9; // rsi

  v3 = a1[5];
  v4 = *(_QWORD **)(v3 + 792);
  if ( v4 != *(_QWORD **)(*(_QWORD *)(v3 + 784) + 32LL) )
  {
    v5 = *v4 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v5 )
      BUG();
    v6 = *(_QWORD *)v5;
    if ( (*(_QWORD *)v5 & 4) == 0 && (*(_BYTE *)(v5 + 46) & 4) != 0 )
    {
      while ( 1 )
      {
        v5 = v6 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)((v6 & 0xFFFFFFFFFFFFFFF8LL) + 46) & 4) == 0 )
          break;
        v6 = *(_QWORD *)v5;
      }
    }
    a1[18] = v5;
  }
  *(_QWORD *)(v3 + 792) = *a2;
  result = (unsigned __int64)(a2 + 1);
  if ( a1 + 10 != a2 + 1 )
  {
    v8 = a1[10];
    if ( v8 )
      result = sub_161E7C0((__int64)(a1 + 10), v8);
    v9 = a2[1];
    a1[10] = v9;
    if ( v9 )
      return sub_1623A60((__int64)(a1 + 10), v9, 2);
  }
  return result;
}
