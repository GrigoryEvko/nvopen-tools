// Function: sub_28ED760
// Address: 0x28ed760
//
_QWORD *__fastcall sub_28ED760(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // r8
  __int64 v12; // rsi
  signed __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // rcx
  __int64 v17; // rcx

  v5 = a2[9];
  v6 = a2[6];
  v7 = a2[7];
  v8 = a2[2];
  v9 = a2[4];
  v10 = a2[8];
  v11 = a2[3];
  v12 = a2[5];
  v13 = 0xAAAAAAAAAAAAAAABLL * ((v9 - v8) >> 3) + 0xAAAAAAAAAAAAAAABLL * ((v6 - v7) >> 3) + 21 * (((v5 - v12) >> 3) - 1);
  v14 = v13 >> 2;
  if ( v13 >> 2 > 0 )
  {
    v15 = *(_QWORD *)(a3 + 16);
    while ( *(_QWORD *)(v8 + 16) != v15 )
    {
      v8 += 24;
      if ( v9 == v8 )
      {
        v8 = *(_QWORD *)(v12 + 8);
        v12 += 8;
        v9 = v8 + 504;
        v11 = v8;
        if ( v15 == *(_QWORD *)(v8 + 16) )
          goto LABEL_15;
      }
      else if ( v15 == *(_QWORD *)(v8 + 16) )
      {
        goto LABEL_15;
      }
      v8 += 24;
      if ( v9 == v8 )
      {
        v8 = *(_QWORD *)(v12 + 8);
        v12 += 8;
        v9 = v8 + 504;
        v11 = v8;
      }
      if ( v15 == *(_QWORD *)(v8 + 16) )
        break;
      v8 += 24;
      if ( v9 == v8 )
      {
        v8 = *(_QWORD *)(v12 + 8);
        v12 += 8;
        v9 = v8 + 504;
        v11 = v8;
      }
      if ( v15 == *(_QWORD *)(v8 + 16) )
        break;
      v8 += 24;
      if ( v9 == v8 )
      {
        v8 = *(_QWORD *)(v12 + 8);
        v12 += 8;
        v9 = v8 + 504;
        v11 = v8;
        if ( !--v14 )
        {
LABEL_17:
          v13 = 21 * (((v5 - v12) >> 3) - 1)
              - 0x5555555555555555LL * ((v6 - v7) >> 3)
              - 0x5555555555555555LL * ((v9 - v8) >> 3);
          goto LABEL_18;
        }
      }
      else if ( !--v14 )
      {
        goto LABEL_17;
      }
    }
    goto LABEL_15;
  }
LABEL_18:
  switch ( v13 )
  {
    case 2LL:
      v17 = *(_QWORD *)(a3 + 16);
LABEL_29:
      if ( v17 == *(_QWORD *)(v8 + 16) )
        goto LABEL_15;
      v8 += 24;
      if ( v9 == v8 )
      {
        v8 = *(_QWORD *)(v12 + 8);
        v12 += 8;
        v9 = v8 + 504;
        v11 = v8;
      }
      goto LABEL_26;
    case 3LL:
      v17 = *(_QWORD *)(a3 + 16);
      if ( *(_QWORD *)(v8 + 16) == v17 )
        goto LABEL_15;
      v8 += 24;
      if ( v9 == v8 )
      {
        v8 = *(_QWORD *)(v12 + 8);
        v12 += 8;
        v9 = v8 + 504;
        v11 = v8;
      }
      goto LABEL_29;
    case 1LL:
      v17 = *(_QWORD *)(a3 + 16);
LABEL_26:
      if ( v17 != *(_QWORD *)(v8 + 16) )
        break;
LABEL_15:
      *a1 = v8;
      a1[1] = v11;
      a1[2] = v9;
      a1[3] = v12;
      return a1;
  }
  *a1 = v6;
  a1[1] = v7;
  a1[2] = v10;
  a1[3] = v5;
  return a1;
}
