// Function: sub_1851200
// Address: 0x1851200
//
__int64 __fastcall sub_1851200(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rsi
  _QWORD *v4; // rax
  _QWORD *v5; // r8
  __int64 v6; // rcx
  __int64 v7; // rdx
  unsigned __int64 v8; // rdx
  _QWORD *v9; // rax
  _QWORD *v10; // r8
  __int64 v11; // rsi
  __int64 v12; // rcx
  unsigned __int64 v13; // rax

  if ( *(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) != *(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 32) )
    return a2;
  v3 = *(_QWORD *)(a2 & 0xFFFFFFFFFFFFFFF8LL);
  v4 = *(_QWORD **)(a1 + 144);
  if ( !v4 )
    return 0;
  v5 = (_QWORD *)(a1 + 136);
  do
  {
    while ( 1 )
    {
      v6 = v4[2];
      v7 = v4[3];
      if ( v3 <= v4[4] )
        break;
      v4 = (_QWORD *)v4[3];
      if ( !v7 )
        goto LABEL_8;
    }
    v5 = v4;
    v4 = (_QWORD *)v4[2];
  }
  while ( v6 );
LABEL_8:
  if ( (_QWORD *)(a1 + 136) == v5 )
    return 0;
  if ( v3 < v5[4] )
    return 0;
  v8 = v5[5];
  if ( !v8 )
    return 0;
  v9 = *(_QWORD **)(a1 + 16);
  if ( v9 )
  {
    v10 = (_QWORD *)(a1 + 8);
    do
    {
      while ( 1 )
      {
        v11 = v9[2];
        v12 = v9[3];
        if ( v8 <= v9[4] )
          break;
        v9 = (_QWORD *)v9[3];
        if ( !v12 )
          goto LABEL_16;
      }
      v10 = v9;
      v9 = (_QWORD *)v9[2];
    }
    while ( v11 );
LABEL_16:
    v13 = 0;
    if ( (_QWORD *)(a1 + 8) != v10 && v8 >= v10[4] )
      v13 = (unsigned __int64)(v10 + 4) & 0xFFFFFFFFFFFFFFFBLL;
  }
  else
  {
    v13 = 0;
  }
  return (4LL * *(unsigned __int8 *)(a1 + 178)) | v13;
}
