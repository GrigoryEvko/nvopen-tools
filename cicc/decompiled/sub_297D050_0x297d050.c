// Function: sub_297D050
// Address: 0x297d050
//
_QWORD *__fastcall sub_297D050(_QWORD *a1, __int64 *a2, __int64 *a3, __int64 a4, int a5)
{
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 v9; // rsi
  unsigned int v10; // eax
  unsigned int v11; // ecx
  __int64 v12; // rdx
  _QWORD *v13; // r13
  __int64 v14; // rax
  __int64 v15; // rsi
  unsigned int v16; // eax
  _BYTE *v17; // rdx
  _BYTE *v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rax
  _QWORD *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // [rsp+8h] [rbp-68h]
  __int64 v26; // [rsp+10h] [rbp-60h]
  __m128i v28; // [rsp+20h] [rbp-50h] BYREF
  __int64 v29; // [rsp+30h] [rbp-40h]

  v7 = *a3;
  v8 = *a3 + 8LL * *((unsigned int *)a3 + 2);
  if ( *a3 == v8 )
  {
LABEL_19:
    *a1 = 0;
    a1[1] = 0;
    a1[2] = 0;
    return a1;
  }
  while ( 1 )
  {
    v19 = *(_QWORD *)(*a2 + 8);
    v20 = *(_QWORD *)(*(_QWORD *)(a4 + 32) + 40LL);
    if ( v20 )
    {
      v9 = (unsigned int)(*(_DWORD *)(v20 + 44) + 1);
      v10 = *(_DWORD *)(v20 + 44) + 1;
    }
    else
    {
      v9 = 0;
      v10 = 0;
    }
    v11 = *(_DWORD *)(v19 + 32);
    if ( v10 >= v11 )
      goto LABEL_15;
    v12 = *(_QWORD *)(v19 + 24);
    if ( !*(_QWORD *)(v12 + 8 * v9) )
      goto LABEL_15;
    v13 = *(_QWORD **)(v8 - 8);
    v14 = *(_QWORD *)(v13[4] + 40LL);
    if ( v14 )
    {
      v15 = (unsigned int)(*(_DWORD *)(v14 + 44) + 1);
      v16 = *(_DWORD *)(v14 + 44) + 1;
    }
    else
    {
      v15 = 0;
      v16 = 0;
    }
    if ( v11 <= v16 || !*(_QWORD *)(v12 + 8 * v15) || !sub_297BA30(a4, *(_QWORD *)(v8 - 8), a5) )
      goto LABEL_15;
    if ( a5 == 1 )
    {
      v23 = sub_297C710(a4, (__int64)v13);
    }
    else
    {
      if ( a5 == 3 )
      {
        v17 = (_BYTE *)v13[7];
        v18 = *(_BYTE **)(a4 + 56);
      }
      else
      {
        v17 = (_BYTE *)v13[1];
        v18 = *(_BYTE **)(a4 + 8);
      }
      v25 = (__int64)v17;
      if ( !(unsigned __int8)sub_D9B130(*(_QWORD *)(*a2 + 16), v18, v17) )
        goto LABEL_15;
      v22 = sub_DCC810(*(__int64 **)(*a2 + 16), (__int64)v18, v25, 0, 0);
      if ( *((_WORD *)v22 + 12) )
        goto LABEL_15;
      v23 = v22[4];
    }
    if ( v23 )
      break;
LABEL_15:
    v8 -= 8;
    if ( v7 == v8 )
      goto LABEL_19;
  }
  if ( v13[5]
    && *(_DWORD *)a4 != 2
    && (v26 = v23, sub_297CE50(v28.m128i_i64, *a2, a4, (__int64)v13), v23 = v26, v28.m128i_i64[0]) )
  {
    v24 = v29;
    *(__m128i *)a1 = _mm_loadu_si128(&v28);
    a1[2] = v24;
  }
  else
  {
    *a1 = v13;
    *((_DWORD *)a1 + 2) = a5;
    a1[2] = v23;
  }
  return a1;
}
