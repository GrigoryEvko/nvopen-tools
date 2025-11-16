// Function: sub_222DA10
// Address: 0x222da10
//
__int64 __fastcall sub_222DA10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  long double v6; // fst7
  long double v7; // fst6
  long double v8; // fst6
  long double v9; // fst5
  long double v10; // rt0
  long double v11; // fst5
  long double v12; // fst6
  long double v13; // rt1
  long double v15; // fst6
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rsi
  long double v20; // rt0
  long double v21; // fst5

  v4 = *(_QWORD *)(a1 + 8);
  v5 = a4 + a3;
  if ( v5 <= v4 )
    return 0;
  if ( v4 )
  {
    v6 = *(float *)a1;
    v7 = (long double)v5;
    if ( v5 >= 0 )
      goto LABEL_4;
  }
  else
  {
    v6 = *(float *)a1;
    if ( (unsigned __int64)v5 < 0xB )
      v5 = 11;
    v7 = (long double)v5;
    if ( v5 >= 0 )
    {
LABEL_4:
      v8 = v7 / v6;
      v9 = (long double)a2;
      if ( a2 >= 0 )
      {
        v10 = v9;
        v11 = v8;
        v12 = v10;
        goto LABEL_7;
      }
      goto LABEL_22;
    }
  }
  v8 = (v7 + 1.8446744e19) / v6;
  v9 = (long double)a2;
  if ( a2 >= 0 )
  {
    v13 = v9;
    v11 = v8;
    v12 = v13;
LABEL_7:
    if ( v11 >= v12 )
    {
      _FST7 = v11;
LABEL_9:
      __asm { frndint }
      v15 = _FST7 + 1.0;
      if ( v15 >= 9.223372e18 )
        v16 = (__int64)(v15 - 9.223372e18) ^ 0x8000000000000000LL;
      else
        v16 = (__int64)v15;
      v17 = 2 * a2;
      if ( v16 >= v17 )
        v17 = v16;
      sub_222D860(a1, v17);
      return 1;
    }
    goto LABEL_15;
  }
LABEL_22:
  v20 = v9 + 1.8446744e19;
  v21 = v8;
  v12 = v20;
  if ( v21 >= v20 )
  {
    _FST7 = v21;
    goto LABEL_9;
  }
LABEL_15:
  _FST7 = v6 * v12;
  __asm { frndint }
  if ( _FST7 >= 9.223372e18 )
  {
    *(_QWORD *)(a1 + 8) = (__int64)(_FST7 - 9.223372e18);
    *(_QWORD *)(a1 + 8) ^= 0x8000000000000000LL;
  }
  else
  {
    *(_QWORD *)(a1 + 8) = (__int64)_FST7;
  }
  return 0;
}
