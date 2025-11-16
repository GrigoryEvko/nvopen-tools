// Function: sub_17E8510
// Address: 0x17e8510
//
__int64 __fastcall sub_17E8510(_QWORD *a1, unsigned __int64 *a2)
{
  unsigned __int64 v2; // r8
  unsigned __int64 v3; // rsi
  _QWORD *v4; // r9
  unsigned __int64 v5; // r10
  _QWORD *v6; // rax
  _QWORD *v7; // rdi
  _QWORD *v8; // rdi
  _QWORD *v9; // rcx

  v2 = *a2;
  v3 = a1[1];
  v4 = *(_QWORD **)(*a1 + 8 * (v2 % v3));
  v5 = v2 % v3;
  if ( !v4 )
    return 0;
  v6 = (_QWORD *)*v4;
  if ( v2 != *(_QWORD *)(*v4 + 8LL) )
  {
    do
    {
      v7 = (_QWORD *)*v6;
      if ( !*v6 )
        return 0;
      v4 = v6;
      if ( v5 != v7[1] % v3 )
        return 0;
      v6 = (_QWORD *)*v6;
    }
    while ( v2 != v7[1] );
  }
  v8 = (_QWORD *)*v4;
  if ( !*v4 )
    return 0;
  v9 = (_QWORD *)*v8;
  if ( *v8 )
  {
    do
    {
      if ( v9[1] % v3 != v5 )
        break;
      if ( v2 != v9[1] )
        break;
      v9 = (_QWORD *)*v9;
    }
    while ( v9 );
  }
  return *v4;
}
