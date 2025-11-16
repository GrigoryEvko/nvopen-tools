// Function: sub_1D18E30
// Address: 0x1d18e30
//
char __fastcall sub_1D18E30(_QWORD *a1, __int64 a2, int a3, int a4)
{
  _DWORD *v5; // r12
  __int64 v8; // rax
  __int16 v9; // dx
  char result; // al
  __int64 v11; // r13
  __int64 v12; // r15
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 v15; // rax
  unsigned int v16; // r12d
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned int v19; // ebx
  __int64 v20; // [rsp+8h] [rbp-68h]
  __m128i v21; // [rsp+10h] [rbp-60h] BYREF
  __int64 v22; // [rsp+20h] [rbp-50h]
  __int64 v23; // [rsp+28h] [rbp-48h]

  v5 = a1;
  v8 = *a1;
  if ( *a1 == a2 )
    goto LABEL_7;
  while ( 1 )
  {
    if ( !a4 )
      return 0;
    v9 = *(_WORD *)(v8 + 24);
    if ( v9 == 2 )
      break;
    if ( v9 != 185 || (*(_BYTE *)(v8 + 26) & 8) != 0 )
      return 0;
    v5 = *(_DWORD **)(v8 + 32);
    --a4;
    v8 = *(_QWORD *)v5;
    if ( *(_QWORD *)v5 == a2 )
    {
LABEL_7:
      if ( v5[2] == a3 )
        return 1;
    }
  }
  v11 = *(_QWORD *)(v8 + 32);
  v12 = 40LL * *(unsigned int *)(v8 + 56);
  v13 = v11 + v12;
  v14 = (__int64)(0xCCCCCCCCCCCCCCCDLL * (v12 >> 3)) >> 2;
  if ( !v14 )
  {
    v20 = *(_QWORD *)(v8 + 32);
LABEL_18:
    v15 = v13 - v20;
    if ( v13 - v20 != 80 )
    {
      if ( v15 != 120 )
      {
        if ( v15 != 40 )
        {
LABEL_21:
          v20 = v13;
          goto LABEL_22;
        }
LABEL_54:
        if ( *(_QWORD *)v20 != a2 || *(_DWORD *)(v20 + 8) != a3 )
          goto LABEL_21;
LABEL_56:
        if ( v13 != v20 )
          goto LABEL_32;
LABEL_22:
        v22 = a2;
        LODWORD(v23) = a3;
        if ( v14 )
          goto LABEL_23;
LABEL_41:
        v18 = v20 - v11;
        if ( v20 - v11 == 80 )
        {
          v19 = a4 - 1;
        }
        else
        {
          if ( v18 != 120 )
          {
            if ( v18 != 40 )
              return 1;
            v19 = a4 - 1;
            goto LABEL_45;
          }
          v19 = a4 - 1;
          v21 = _mm_loadu_si128((const __m128i *)v11);
          if ( !(unsigned __int8)sub_1D18E30(&v21, v22, v23, v19) )
            return v20 == v11;
          v11 += 40;
        }
        v21 = _mm_loadu_si128((const __m128i *)v11);
        if ( !(unsigned __int8)sub_1D18E30(&v21, v22, v23, v19) )
          return v20 == v11;
        v11 += 40;
LABEL_45:
        v21 = _mm_loadu_si128((const __m128i *)v11);
        result = sub_1D18E30(&v21, v22, v23, v19);
        if ( !result )
          return v20 == v11;
        return result;
      }
      if ( *(_QWORD *)v20 == a2 && *(_DWORD *)(v20 + 8) == a3 )
        goto LABEL_56;
      v20 += 40;
    }
    if ( *(_QWORD *)v20 == a2 && *(_DWORD *)(v20 + 8) == a3 )
      goto LABEL_56;
    v20 += 40;
    goto LABEL_54;
  }
  v20 = *(_QWORD *)(v8 + 32);
  while ( *(_QWORD *)v20 != a2 || *(_DWORD *)(v20 + 8) != a3 )
  {
    if ( *(_QWORD *)(v20 + 40) == a2 && *(_DWORD *)(v20 + 48) == a3 )
    {
      v20 += 40;
      break;
    }
    if ( *(_QWORD *)(v20 + 80) == a2 && *(_DWORD *)(v20 + 88) == a3 )
    {
      v20 += 80;
      break;
    }
    if ( *(_QWORD *)(v20 + 120) == a2 && *(_DWORD *)(v20 + 128) == a3 )
    {
      v20 += 120;
      break;
    }
    v20 += 160;
    if ( v11 + 160 * v14 == v20 )
      goto LABEL_18;
  }
  if ( v13 != v20 )
  {
LABEL_32:
    if ( !sub_1D18C00(a2, 1, a3) )
    {
      v11 = *(_QWORD *)(*(_QWORD *)v5 + 32LL);
      v17 = 40LL * *(unsigned int *)(*(_QWORD *)v5 + 56LL);
      v20 = v11 + v17;
      v14 = (__int64)(0xCCCCCCCCCCCCCCCDLL * (v17 >> 3)) >> 2;
      goto LABEL_22;
    }
    return 1;
  }
  v22 = a2;
  LODWORD(v23) = a3;
LABEL_23:
  v16 = a4 - 1;
  while ( 1 )
  {
    v21 = _mm_loadu_si128((const __m128i *)v11);
    if ( !(unsigned __int8)sub_1D18E30(&v21, v22, v23, v16) )
      return v20 == v11;
    v21 = _mm_loadu_si128((const __m128i *)(v11 + 40));
    if ( !(unsigned __int8)sub_1D18E30(&v21, v22, v23, v16) )
      return v20 == v11 + 40;
    v21 = _mm_loadu_si128((const __m128i *)(v11 + 80));
    if ( !(unsigned __int8)sub_1D18E30(&v21, v22, v23, v16) )
      return v20 == v11 + 80;
    v21 = _mm_loadu_si128((const __m128i *)(v11 + 120));
    if ( !(unsigned __int8)sub_1D18E30(&v21, v22, v23, v16) )
      return v20 == v11 + 120;
    v11 += 160;
    if ( !--v14 )
      goto LABEL_41;
  }
}
