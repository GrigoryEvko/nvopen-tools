// Function: sub_1D18CA0
// Address: 0x1d18ca0
//
__int64 __fastcall sub_1D18CA0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  __int64 v6; // r10
  _QWORD *v7; // r9
  __int64 v8; // rdi
  __int64 v9; // r10
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rcx
  __int64 v14; // rcx

  v5 = *(_QWORD *)(a3 + 48);
  if ( v5 )
  {
    v6 = 8 * a2;
    v7 = &a1[a2];
    v8 = (8 * a2) >> 5;
    v9 = v6 >> 3;
    while ( 1 )
    {
      v10 = *(_QWORD *)(v5 + 16);
      if ( v8 > 0 )
      {
        v11 = a1;
        v12 = v8;
        while ( v10 != *v11 )
        {
          if ( v10 == v11[1] )
          {
            ++v11;
            goto LABEL_10;
          }
          if ( v10 == v11[2] )
          {
            v11 += 2;
            goto LABEL_10;
          }
          if ( v10 == v11[3] )
          {
            v11 += 3;
            goto LABEL_10;
          }
          v11 += 4;
          if ( !--v12 )
          {
            v14 = v7 - v11;
            goto LABEL_17;
          }
        }
        goto LABEL_10;
      }
      v14 = v9;
      v11 = a1;
LABEL_17:
      if ( v14 == 2 )
        goto LABEL_21;
      if ( v14 == 3 )
        break;
      if ( v14 != 1 )
        return 0;
LABEL_23:
      if ( v10 != *v11 )
        return 0;
LABEL_10:
      if ( v7 == v11 )
        return 0;
      v5 = *(_QWORD *)(v5 + 32);
      if ( !v5 )
        return 1;
    }
    if ( v10 == *v11 )
      goto LABEL_10;
    ++v11;
LABEL_21:
    if ( v10 == *v11 )
      goto LABEL_10;
    ++v11;
    goto LABEL_23;
  }
  return 0;
}
