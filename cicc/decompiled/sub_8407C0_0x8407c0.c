// Function: sub_8407C0
// Address: 0x8407c0
//
__int64 __fastcall sub_8407C0(
        __m128i *a1,
        const __m128i *a2,
        _BOOL4 a3,
        unsigned int a4,
        unsigned int a5,
        __int64 a6,
        int a7,
        int a8,
        __int64 a9,
        __m128i *a10,
        _DWORD *a11)
{
  __int64 v13; // r15
  const __m128i *v14; // rcx
  unsigned int j; // ebx
  __int64 v16; // r8
  unsigned int v17; // r14d
  int v19; // edx
  __int32 *v20; // rsi
  _QWORD *v21; // r12
  __int64 *k; // r12
  __int64 *v23; // rbx
  unsigned int v24; // esi
  __int8 v25; // dl
  const __m128i *v26; // rax
  char v27; // dl
  __int64 v28; // rax
  int v29; // eax
  __int64 v30; // rcx
  __int64 v31; // r8
  int v32; // eax
  int v33; // eax
  const __m128i *v34; // r8
  unsigned int v35; // eax
  __int64 i; // rsi
  int v37; // eax
  __int64 v39; // [rsp+8h] [rbp-58h]
  __int64 v40; // [rsp+8h] [rbp-58h]
  __int64 v41; // [rsp+8h] [rbp-58h]
  __int64 v43; // [rsp+10h] [rbp-50h]
  __int64 v44; // [rsp+10h] [rbp-50h]
  __int64 v45; // [rsp+10h] [rbp-50h]
  int v47; // [rsp+18h] [rbp-48h]
  const __m128i *v48; // [rsp+18h] [rbp-48h]
  unsigned int v49; // [rsp+24h] [rbp-3Ch] BYREF
  __int64 *v50[7]; // [rsp+28h] [rbp-38h] BYREF

  *a11 = 0;
  *(_OWORD *)a9 = 0;
  *(_OWORD *)(a9 + 16) = 0;
  *(_OWORD *)(a9 + 32) = 0;
  v13 = a1->m128i_i64[0];
  v49 = 0;
  if ( a3 || !(unsigned int)sub_8D3A70(a2) )
  {
    if ( !(unsigned int)sub_8D3A70(v13) )
    {
      if ( !(unsigned int)sub_8D3D40(v13) )
      {
        v17 = sub_8D3D40(a2);
        if ( !v17 )
        {
          v14 = a2;
          j = 0;
          v16 = 0;
          goto LABEL_7;
        }
      }
      *(_BYTE *)(a9 + 17) |= 1u;
LABEL_6:
      v14 = a2;
      j = 0;
      v16 = 0;
      v17 = 1;
      goto LABEL_7;
    }
    v19 = sub_840360(a1->m128i_i64, (__int64)a2, 0, a3, a4, a5, a6, a7, a8, a9, &v49, (const __m128i **)v50);
    if ( v19 )
      goto LABEL_6;
    if ( !dword_4D0478C || (a8 & 0x1000000) == 0 || (v33 = sub_8D3BB0(a2), v19 = 0, !v33) )
    {
      v17 = v49;
      *a11 = 1;
      if ( !v17 )
      {
        v14 = a2;
        v24 = 0;
        j = 413;
        v16 = 0;
        goto LABEL_22;
      }
      goto LABEL_12;
    }
    if ( v49 )
    {
      *a11 = 1;
LABEL_12:
      v14 = a2;
      v17 = 0;
      j = 417;
      v16 = 0;
      goto LABEL_13;
    }
    goto LABEL_45;
  }
  if ( (a8 & 0x80u) != 0
    && dword_4D04474
    && a1[1].m128i_i8[0] == 1
    && (a1[1].m128i_i8[4] & 0x10) == 0
    && (unsigned int)sub_837960(a1, a2, a4, a5, a8, (__int64 *)a9, a10)
    || (unsigned int)sub_836C50(a1, 0, a2, 1u, a4, a5, a6, a7, a8, a9, a10, &v49, v50) )
  {
    goto LABEL_6;
  }
  if ( dword_4D0478C && (a8 & 0x1000000) != 0 && (unsigned int)sub_8D3BB0(a2) && !v49 )
  {
LABEL_45:
    j = 0;
    v16 = 0;
    v24 = 0;
    goto LABEL_46;
  }
  v34 = a2;
  for ( *a11 = 1; v34[8].m128i_i8[12] == 12; v34 = (const __m128i *)v34[10].m128i_i64[0] )
    ;
  v48 = v34;
  v35 = sub_8D3A70(v13);
  v16 = (__int64)v48;
  v17 = v35;
  if ( v35 )
  {
    for ( i = v13; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( (const __m128i *)i == v48 || (v37 = sub_8DED30(v48, i, 1), v16 = (__int64)v48, (v19 = v37) != 0) )
    {
      if ( v49 )
      {
        j = 290;
        v19 = 1;
LABEL_65:
        if ( *a11 )
        {
          v14 = a2;
          v17 = 0;
          goto LABEL_13;
        }
        return 0;
      }
      j = 334;
      v24 = 1;
    }
    else
    {
      v24 = v49;
      j = 312;
      if ( v49 )
      {
        j = 348;
        goto LABEL_65;
      }
    }
LABEL_46:
    if ( *a11 )
    {
      v14 = a2;
      v17 = 0;
      goto LABEL_22;
    }
    return 0;
  }
  v14 = a2;
  for ( j = 415 - ((v49 == 0) - 1); v14[8].m128i_i8[12] == 12; v14 = (const __m128i *)v14[10].m128i_i64[0] )
    ;
LABEL_7:
  if ( *a11 )
  {
    v24 = v49;
    v19 = 0;
    if ( !v49 )
    {
LABEL_22:
      v25 = a2[8].m128i_i8[12];
      if ( v25 == 12 )
      {
        v26 = a2;
        do
        {
          v26 = (const __m128i *)v26[10].m128i_i64[0];
          v25 = v26[8].m128i_i8[12];
        }
        while ( v25 == 12 );
      }
      if ( v25 )
      {
        v27 = *(_BYTE *)(v13 + 140);
        if ( v27 == 12 )
        {
          v28 = v13;
          do
          {
            v28 = *(_QWORD *)(v28 + 160);
            v27 = *(_BYTE *)(v28 + 140);
          }
          while ( v27 == 12 );
        }
        if ( v27 )
        {
          v40 = v16;
          v44 = (__int64)v14;
          v29 = sub_8D23B0(a2);
          v30 = v44;
          v31 = v40;
          if ( v29 && (v32 = sub_8D3A70(a2), v30 = v44, v31 = v40, v32) )
          {
            if ( (unsigned int)sub_6E5430() )
              sub_685360(0x203u, &a1[4].m128i_i32[1], v44);
          }
          else
          {
            v41 = v31;
            v45 = v30;
            if ( (unsigned int)sub_6E5430() )
            {
              if ( v24 )
                sub_685360(j, &a1[4].m128i_i32[1], v41);
              else
                sub_6E6960(j, (__int64)a1, v13, v45);
            }
          }
        }
      }
LABEL_20:
      sub_6E6840((__int64)a1);
      return v17;
    }
LABEL_13:
    v39 = v16;
    v43 = (__int64)v14;
    v47 = v19;
    if ( v50[0] )
    {
      if ( (unsigned int)sub_6E5430() )
      {
        v20 = &a1[4].m128i_i32[1];
        if ( v47 )
          v21 = sub_67DA80(j, v20, v39);
        else
          v21 = sub_67DAA0(j, v20, v13, v43);
        sub_82E650(v50[0], 0, 0, 0, v21);
        sub_685910((__int64)v21, 0);
      }
      for ( k = v50[0]; k; qword_4D03C68 = v23 )
      {
        v23 = k;
        k = (__int64 *)*k;
        sub_725130((__int64 *)v23[5]);
        sub_82D8A0((_QWORD *)v23[15]);
        *v23 = (__int64)qword_4D03C68;
      }
    }
    goto LABEL_20;
  }
  return v17;
}
