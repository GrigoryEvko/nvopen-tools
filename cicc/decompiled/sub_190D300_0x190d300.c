// Function: sub_190D300
// Address: 0x190d300
//
__int64 __fastcall sub_190D300(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *v3; // rdi
  _QWORD *v4; // rbx
  unsigned __int64 v5; // rdx
  _QWORD *v6; // rax
  _QWORD *v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // r12

  v3 = *(_QWORD **)(a1 + 16);
  v4 = (_QWORD *)(a1 + 8);
  if ( !v3 )
    return 0;
  v5 = *a2;
  while ( 1 )
  {
    while ( v3[4] < v5 )
    {
      v3 = (_QWORD *)v3[3];
      if ( !v3 )
        return 0;
    }
    v6 = (_QWORD *)v3[2];
    if ( v3[4] <= v5 )
      break;
    v4 = v3;
    v3 = (_QWORD *)v3[2];
    if ( !v6 )
      return 0;
  }
  v8 = (_QWORD *)v3[3];
  if ( v8 )
  {
    do
    {
      while ( 1 )
      {
        v9 = v8[2];
        v10 = v8[3];
        if ( v8[4] > v5 )
          break;
        v8 = (_QWORD *)v8[3];
        if ( !v10 )
          goto LABEL_13;
      }
      v4 = v8;
      v8 = (_QWORD *)v8[2];
    }
    while ( v9 );
  }
LABEL_13:
  while ( v6 )
  {
    while ( 1 )
    {
      v11 = v6[3];
      if ( v6[4] >= v5 )
        break;
      v6 = (_QWORD *)v6[3];
      if ( !v11 )
        goto LABEL_16;
    }
    v3 = v6;
    v6 = (_QWORD *)v6[2];
  }
LABEL_16:
  if ( v4 == v3 )
    return 0;
  v12 = 0;
  do
  {
    ++v12;
    v3 = (_QWORD *)sub_220EF30(v3);
  }
  while ( v3 != v4 );
  return v12;
}
