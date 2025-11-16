// Function: sub_29B9A00
// Address: 0x29b9a00
//
__int64 *__fastcall sub_29B9A00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r15
  __int64 i; // r8
  __int64 *result; // rax
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 *v11; // r10
  __int64 v12; // rsi
  __int64 v13; // r11
  __int64 v14; // r11
  double v15; // xmm0_8
  __int64 v16; // r11
  double v17; // xmm1_8
  double v18; // xmm2_8
  double v19; // xmm0_8
  __int64 v20; // rsi
  __int64 v21; // r8
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rdx
  double v25; // xmm0_8
  __int64 v26; // rdx
  double v27; // xmm1_8
  double v28; // xmm2_8
  double v29; // xmm0_8
  __int64 v30; // r12
  __int64 v31; // r12
  __int64 v32; // r9
  __int64 v33; // r9
  __int64 v35; // [rsp+8h] [rbp-30h]

  v5 = (a3 - 1) / 2;
  v35 = a3 & 1;
  if ( a2 < v5 )
  {
    for ( i = a2; ; i = v10 )
    {
      v10 = 2 * (i + 1);
      result = (__int64 *)(a1 + 16 * (i + 1));
      v9 = *result;
      v11 = (__int64 *)(a1 + 8 * (v10 - 1));
      v12 = *v11;
      v13 = ***(_QWORD ***)(*result + 32);
      if ( (v13 == 0) != (***(_QWORD ***)(*v11 + 32) == 0) )
        break;
      v14 = *(_QWORD *)(v12 + 24);
      if ( v14 < 0 )
      {
        v31 = *(_QWORD *)(v12 + 24) & 1LL | (*(_QWORD *)(v12 + 24) >> 1);
        v15 = (double)(int)v31 + (double)(int)v31;
      }
      else
      {
        v15 = (double)(int)v14;
      }
      v16 = *(_QWORD *)(v9 + 24);
      v17 = *(double *)(v12 + 16) / v15;
      if ( v16 < 0 )
      {
        v30 = *(_QWORD *)(v9 + 24) & 1LL | (*(_QWORD *)(v9 + 24) >> 1);
        v18 = (double)(int)v30 + (double)(int)v30;
      }
      else
      {
        v18 = (double)(int)v16;
      }
      v19 = *(double *)(v9 + 16) / v18;
      if ( v19 > v17 )
      {
        result = (__int64 *)(a1 + 8 * (v10 - 1));
        v9 = *v11;
        --v10;
        goto LABEL_5;
      }
      if ( v17 > v19 )
        goto LABEL_5;
      if ( *(_QWORD *)v12 > *(_QWORD *)v9 )
      {
        v9 = *v11;
        --v10;
        result = v11;
      }
      *(_QWORD *)(a1 + 8 * i) = v9;
      if ( v10 >= v5 )
      {
LABEL_17:
        if ( v35 )
          goto LABEL_18;
        goto LABEL_38;
      }
LABEL_6:
      ;
    }
    if ( !v13 )
    {
      result = (__int64 *)(a1 + 8 * (v10 - 1));
      v9 = *v11;
      --v10;
    }
LABEL_5:
    *(_QWORD *)(a1 + 8 * i) = v9;
    if ( v10 >= v5 )
      goto LABEL_17;
    goto LABEL_6;
  }
  result = (__int64 *)(a1 + 8 * a2);
  if ( (a3 & 1) != 0 )
    goto LABEL_32;
  v10 = a2;
LABEL_38:
  if ( (a3 - 2) / 2 == v10 )
  {
    v10 = 2 * v10 + 1;
    *result = *(_QWORD *)(a1 + 8 * v10);
    result = (__int64 *)(a1 + 8 * v10);
  }
LABEL_18:
  v20 = (v10 - 1) / 2;
  if ( v10 > a2 )
  {
    v21 = v10;
    while ( 1 )
    {
      result = (__int64 *)(a1 + 8 * v20);
      v22 = *result;
      v23 = ***(_QWORD ***)(*result + 32);
      if ( (v23 == 0) == (***(_QWORD ***)(a4 + 32) == 0) )
      {
        v24 = *(_QWORD *)(a4 + 24);
        if ( v24 < 0 )
        {
          v33 = *(_QWORD *)(a4 + 24) & 1LL | (*(_QWORD *)(a4 + 24) >> 1);
          v25 = (double)(int)v33 + (double)(int)v33;
        }
        else
        {
          v25 = (double)(int)v24;
        }
        v26 = *(_QWORD *)(v22 + 24);
        v27 = *(double *)(a4 + 16) / v25;
        if ( v26 < 0 )
        {
          v32 = *(_QWORD *)(v22 + 24) & 1LL | (*(_QWORD *)(v22 + 24) >> 1);
          v28 = (double)(int)v32 + (double)(int)v32;
        }
        else
        {
          v28 = (double)(int)v26;
        }
        v29 = *(double *)(v22 + 16) / v28;
        if ( v29 <= v27 && (v27 > v29 || *(_QWORD *)a4 <= *(_QWORD *)v22) )
        {
LABEL_31:
          result = (__int64 *)(a1 + 8 * v21);
          break;
        }
      }
      else if ( v23 )
      {
        goto LABEL_31;
      }
      *(_QWORD *)(a1 + 8 * v21) = v22;
      v21 = v20;
      if ( a2 >= v20 )
        break;
      v20 = (v20 - 1) / 2;
    }
  }
LABEL_32:
  *result = a4;
  return result;
}
