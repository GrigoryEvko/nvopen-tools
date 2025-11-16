// Function: sub_F08550
// Address: 0xf08550
//
__int64 __fastcall sub_F08550(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // r9
  unsigned int v9; // edx
  __int64 v10; // rbx
  __int64 v12; // rdx

  v6 = a4 - a2;
  v7 = (a4 - a2) >> 2;
  if ( v7 > 0 )
  {
    v8 = *(_QWORD *)(a1 - 8);
    v9 = 2 * a2 + 2;
    v10 = a2 + 4 * v7;
    while ( *(_QWORD *)(v8 + 32LL * v9) != a5 )
    {
      if ( *(_QWORD *)(v8 + 32LL * (v9 + 2)) == a5 )
        break;
      if ( *(_QWORD *)(v8 + 32LL * (v9 + 4)) == a5 )
        break;
      a2 += 4;
      if ( *(_QWORD *)(v8 + 32LL * (unsigned int)(2 * a2)) == a5 )
        break;
      v9 += 8;
      if ( v10 == a2 )
      {
        v6 = a4 - a2;
        goto LABEL_10;
      }
    }
    return a1;
  }
LABEL_10:
  switch ( v6 )
  {
    case 2LL:
      v12 = *(_QWORD *)(a1 - 8);
LABEL_20:
      LODWORD(a2) = a2 + 1;
      if ( *(_QWORD *)(v12 + 32LL * (unsigned int)(2 * a2)) == a5 )
        return a1;
      goto LABEL_17;
    case 3LL:
      LODWORD(a2) = a2 + 1;
      v12 = *(_QWORD *)(a1 - 8);
      if ( *(_QWORD *)(v12 + 32LL * (unsigned int)(2 * a2)) == a5 )
        return a1;
      goto LABEL_20;
    case 1LL:
      v12 = *(_QWORD *)(a1 - 8);
LABEL_17:
      if ( *(_QWORD *)(v12 + 32LL * (unsigned int)(2 * a2 + 2)) != a5 )
        return a3;
      return a1;
  }
  return a3;
}
