// Function: sub_7790A0
// Address: 0x7790a0
//
__int64 __fastcall sub_7790A0(__int64 a1, __m128i *a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // r13
  __int64 result; // rax
  unsigned __int64 v7; // r15
  unsigned __int64 v8; // rbx
  __int64 n; // rdx
  unsigned int v10; // ecx
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 *v13; // rsi
  unsigned __int64 v14; // rbx
  char m; // cl
  unsigned __int64 v16; // r10
  unsigned __int64 v17; // r9
  unsigned __int64 v18; // r15
  unsigned __int64 v19; // r15
  char i; // al
  __int64 v21; // rbx
  unsigned int v22; // ebx
  __int64 v23; // rbx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rbx
  __int64 j; // r15
  unsigned int v27; // edx
  __int64 v28; // rax
  __int64 v29; // rax
  char k; // dl
  unsigned int v31; // edx
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 *v34; // rbx
  unsigned __int64 v35; // [rsp+8h] [rbp-58h]
  unsigned __int64 v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+10h] [rbp-50h]
  __int64 v38; // [rsp+18h] [rbp-48h]
  __int64 v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+18h] [rbp-48h]
  unsigned __int64 v41; // [rsp+18h] [rbp-48h]
  _DWORD v42[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v5 = (__int64 *)a2;
  switch ( *(_BYTE *)(a3 + 140) )
  {
    case 2:
      *a2 = _mm_load_si128((const __m128i *)&xmmword_4F08290);
      goto LABEL_3;
    case 3:
    case 4:
      *a2 = _mm_load_si128(&xmmword_4F081A0[*(unsigned __int8 *)(a3 + 160)]);
      goto LABEL_3;
    case 5:
      *a2 = _mm_load_si128(&xmmword_4F081A0[*(unsigned __int8 *)(a3 + 160)]);
      a2[1] = _mm_load_si128(&xmmword_4F081A0[*(unsigned __int8 *)(a3 + 160)]);
      goto LABEL_3;
    case 6:
    case 0x13:
      *a2 = 0;
      a2[1] = 0;
      goto LABEL_3;
    case 8:
      v19 = *(_QWORD *)(a3 + 160);
      for ( i = *(_BYTE *)(v19 + 140); i == 12; i = *(_BYTE *)(v19 + 140) )
        v19 = *(_QWORD *)(v19 + 160);
      v21 = *(_QWORD *)(a3 + 176);
      v42[0] = 1;
      v40 = v21;
      v22 = 16;
      if ( (unsigned __int8)(i - 2) <= 1u || (v22 = sub_7764B0(a1, v19, v42), (result = v42[0]) != 0) )
      {
        result = v22;
        v23 = 0;
        v37 = result;
        if ( v40 )
        {
          do
          {
            sub_7790A0(a1, v5, v19, a4);
            result = (unsigned int)*(unsigned __int8 *)(v19 + 140) - 9;
            if ( (unsigned __int8)(*(_BYTE *)(v19 + 140) - 9) <= 2u )
              *v5 = 0;
            v5 = (__int64 *)((char *)v5 + v37);
            ++v23;
          }
          while ( v40 != v23 );
        }
      }
      return result;
    case 9:
    case 0xA:
      v7 = **(_QWORD **)(a3 + 168);
      v8 = sub_76FF70(*(_QWORD *)(a3 + 160));
      if ( !v8 )
        goto LABEL_17;
      break;
    case 0xB:
      v24 = sub_76FF70(*(_QWORD *)(a3 + 160));
      v25 = v24;
      if ( !v24 )
      {
        a2->m128i_i64[0] = 0;
        goto LABEL_3;
      }
      for ( j = *(_QWORD *)(v24 + 120); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      v27 = qword_4F08388 & (v24 >> 3);
      while ( 2 )
      {
        v28 = qword_4F08380 + 16LL * v27;
        if ( v25 == *(_QWORD *)v28 )
        {
          a2 = (__m128i *)((char *)a2 + *(unsigned int *)(v28 + 8));
        }
        else if ( *(_QWORD *)v28 )
        {
          v27 = qword_4F08388 & (v27 + 1);
          continue;
        }
        break;
      }
      sub_7790A0(a1, a2, j, a4);
      if ( (unsigned __int8)(*(_BYTE *)(j + 140) - 9) <= 2u )
        a2->m128i_i64[0] = 0;
      *v5 = v25;
      goto LABEL_3;
    case 0xD:
      v29 = *(_QWORD *)(a3 + 168);
      for ( k = *(_BYTE *)(v29 + 140); k == 12; k = *(_BYTE *)(v29 + 140) )
        v29 = *(_QWORD *)(v29 + 160);
      if ( k == 7 )
        a2->m128i_i8[0] |= 1u;
      else
        a2->m128i_i8[0] &= ~1u;
      a2->m128i_i64[1] = 0;
      a2->m128i_i8[0] &= ~2u;
      a2->m128i_i32[1] = 0;
      goto LABEL_3;
    case 0xF:
      v14 = *(_QWORD *)(a3 + 160);
      for ( m = *(_BYTE *)(v14 + 140); m == 12; m = *(_BYTE *)(v14 + 140) )
        v14 = *(_QWORD *)(v14 + 160);
      v16 = *(_QWORD *)(a3 + 128);
      v17 = *(_QWORD *)(v14 + 128);
      v42[0] = 1;
      v36 = v16 / v17;
      LODWORD(result) = 16;
      if ( (unsigned __int8)(m - 2) > 1u )
      {
        v35 = v17;
        v41 = v16;
        LODWORD(result) = sub_7764B0(a1, v14, v42);
        v17 = v35;
        v16 = v41;
      }
      result = (unsigned int)result;
      v18 = 0;
      v39 = (unsigned int)result;
      if ( v16 >= v17 )
      {
        do
        {
          result = sub_7790A0(a1, v5, v14, a4);
          ++v18;
          v5 = (__int64 *)((char *)v5 + v39);
        }
        while ( v36 > v18 );
      }
      return result;
    case 0x14:
      a2->m128i_i8[0] = 0;
      a2->m128i_i64[1] = 0;
      a2[1].m128i_i32[0] = 0;
      goto LABEL_3;
    default:
      sub_721090();
  }
  do
  {
    for ( n = *(_QWORD *)(v8 + 120); *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
      ;
    v10 = qword_4F08388 & (v8 >> 3);
    v11 = qword_4F08380 + 16LL * v10;
    v12 = *(_QWORD *)v11;
    if ( v8 == *(_QWORD *)v11 )
    {
LABEL_55:
      v13 = (__int64 *)((char *)v5 + *(unsigned int *)(v11 + 8));
    }
    else
    {
      while ( v12 )
      {
        v10 = qword_4F08388 & (v10 + 1);
        v11 = qword_4F08380 + 16LL * v10;
        v12 = *(_QWORD *)v11;
        if ( *(_QWORD *)v11 == v8 )
          goto LABEL_55;
      }
      v13 = v5;
    }
    v38 = n;
    sub_7790A0(a1, v13, n, a4);
    if ( (unsigned __int8)(*(_BYTE *)(v38 + 140) - 9) <= 2u )
      *v13 = 0;
    v8 = sub_76FF70(*(_QWORD *)(v8 + 112));
  }
  while ( v8 );
LABEL_17:
  while ( v7 )
  {
    if ( (*(_BYTE *)(v7 + 96) & 3) != 0 )
    {
      v31 = qword_4F08388 & (v7 >> 3);
      v32 = qword_4F08380 + 16LL * v31;
      v33 = *(_QWORD *)v32;
      if ( v7 == *(_QWORD *)v32 )
      {
LABEL_65:
        v34 = (__int64 *)((char *)v5 + *(unsigned int *)(v32 + 8));
      }
      else
      {
        while ( v33 )
        {
          v31 = qword_4F08388 & (v31 + 1);
          v32 = qword_4F08380 + 16LL * v31;
          v33 = *(_QWORD *)v32;
          if ( *(_QWORD *)v32 == v7 )
            goto LABEL_65;
        }
        v34 = v5;
      }
      sub_7790A0(a1, v34, *(_QWORD *)(v7 + 40), a4);
      *v34 = v7;
    }
    v7 = *(_QWORD *)v7;
  }
LABEL_3:
  result = -(((unsigned int)((_DWORD)v5 - a4) >> 3) + 10);
  *(_BYTE *)(a4 + result) |= 1 << (((_BYTE)v5 - a4) & 7);
  return result;
}
