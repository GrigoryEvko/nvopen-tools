// Function: sub_15FEEB0
// Address: 0x15feeb0
//
__int64 __fastcall sub_15FEEB0(int a1, __int16 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // r15
  _QWORD *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rsi
  _QWORD *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rsi
  _QWORD *v18; // rax
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // rsi
  _QWORD *v22; // rax
  __int64 v23; // r12
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // [rsp+0h] [rbp-40h]
  __int64 v27; // [rsp+0h] [rbp-40h]

  if ( a1 == 51 )
  {
    if ( a6 )
    {
      v10 = sub_1648A60(56, 2);
      if ( !v10 )
        return 0;
      v15 = *(_QWORD **)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
      {
        v27 = v15[4];
        v16 = sub_1643320(*v15);
        v17 = sub_16463B0(v16, v27);
      }
      else
      {
        v17 = sub_1643320(*v15);
      }
      sub_15FEC10(v10, v17, 51, a2, a3, a4, a5, a6);
    }
    else
    {
      v10 = sub_1648A60(56, 2);
      if ( !v10 )
        return 0;
      v22 = *(_QWORD **)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
      {
        v23 = v22[4];
        v24 = sub_1643320(*v22);
        v25 = sub_16463B0(v24, (unsigned int)v23);
      }
      else
      {
        v25 = sub_1643320(*v22);
      }
      sub_15FEC10(v10, v25, 51, a2, a3, a4, a5, 0);
    }
  }
  else
  {
    if ( a6 )
    {
      v10 = sub_1648A60(56, 2);
      if ( v10 )
      {
        v11 = *(_QWORD **)a3;
        if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
        {
          v26 = v11[4];
          v12 = sub_1643320(*v11);
          v13 = sub_16463B0(v12, v26);
        }
        else
        {
          v13 = sub_1643320(*v11);
        }
        sub_15FEC10(v10, v13, 52, a2, a3, a4, a5, a6);
        return v10;
      }
      return 0;
    }
    v10 = sub_1648A60(56, 2);
    if ( !v10 )
      return 0;
    v18 = *(_QWORD **)a3;
    if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
    {
      v19 = v18[4];
      v20 = sub_1643320(*v18);
      v21 = sub_16463B0(v20, (unsigned int)v19);
    }
    else
    {
      v21 = sub_1643320(*v18);
    }
    sub_15FEC10(v10, v21, 52, a2, a3, a4, a5, 0);
  }
  return v10;
}
