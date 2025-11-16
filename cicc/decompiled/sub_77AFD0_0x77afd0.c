// Function: sub_77AFD0
// Address: 0x77afd0
//
__int64 __fastcall sub_77AFD0(__int64 a1, __int64 a2, _QWORD *a3, FILE *a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  int v9; // edx
  __int64 **v10; // rax
  int v11; // r12d
  unsigned __int64 v12; // r14
  __int64 v13; // rax
  char i; // cl
  unsigned int j; // edx
  __int64 v16; // rax
  __int64 v17; // rax
  int v18; // r13d
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rsi
  int v23; // ecx
  const __m128i *v24; // roff
  __int64 v25; // rax
  int v27; // r13d
  __int64 v28; // rax
  __int64 v29; // rbx
  __int64 *v32; // [rsp+10h] [rbp-70h]
  unsigned __int64 v33; // [rsp+18h] [rbp-68h]
  int v34; // [rsp+24h] [rbp-5Ch]
  signed int v35; // [rsp+30h] [rbp-50h]
  unsigned int v37; // [rsp+48h] [rbp-38h] BYREF
  int v38[13]; // [rsp+4Ch] [rbp-34h] BYREF

  v7 = a3[2];
  v37 = 1;
  v35 = v7;
  v8 = sub_72CD60();
  v9 = 16;
  v33 = v8;
  if ( (unsigned __int8)(*(_BYTE *)(v8 + 140) - 2) > 1u )
    v9 = sub_7764B0(a1, v8, &v37);
  v10 = *(__int64 ***)(a2 + 168);
  v38[0] = v9;
  v32 = *v10;
  if ( *v10 )
  {
    if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
      return 0;
    v29 = a1;
LABEL_31:
    sub_6855B0(0xD27u, (FILE *)(v29 + 112), (_QWORD *)(v29 + 96));
    sub_770D30(v29);
    return 0;
  }
  v11 = 0;
  v34 = 0;
  v12 = sub_76FF70(*(_QWORD *)(a2 + 160));
  if ( v12 )
  {
    while ( 2 )
    {
      v13 = *(_QWORD *)(v12 + 120);
      for ( i = *(_BYTE *)(v13 + 140); i == 12; i = *(_BYTE *)(v13 + 140) )
        v13 = *(_QWORD *)(v13 + 160);
      for ( j = qword_4F08388 & (v12 >> 3); ; j = qword_4F08388 & (j + 1) )
      {
        v16 = qword_4F08380 + 16LL * j;
        if ( *(_QWORD *)v16 == v12 )
          break;
        if ( !*(_QWORD *)v16 )
        {
          v17 = 0;
          if ( i == 6 )
            goto LABEL_23;
LABEL_12:
          if ( v11 > 1 || i != 2 )
            goto LABEL_24;
          ++v11;
          v18 = a5 + v17;
          sub_620D80((_WORD *)(a5 + (unsigned int)v17), (unsigned int)v35);
          v19 = -(((unsigned int)(v18 - a6) >> 3) + 10);
          *(_BYTE *)(a6 + v19) |= 1 << ((v18 - a6) & 7);
          goto LABEL_15;
        }
      }
      v17 = *(unsigned int *)(v16 + 8);
      if ( i != 6 )
        goto LABEL_12;
LABEL_23:
      if ( v34 )
        goto LABEL_24;
      v27 = a5 + v17;
      v32 = (__int64 *)(a5 + v17);
      if ( !sub_777910(a1, v33, v35, 1, a4, a5 + v17, v38) )
        return 0;
      v34 = 1;
      v28 = -(((unsigned int)(v27 - a6) >> 3) + 10);
      *(_BYTE *)(a6 + v28) |= 1 << ((v27 - a6) & 7);
LABEL_15:
      v12 = sub_76FF70(*(_QWORD *)(v12 + 112));
      if ( v12 )
        continue;
      break;
    }
    if ( v34 && v11 == 2 )
    {
      *(_BYTE *)(a6 - 9) |= 1u;
      v20 = *v32;
      v21 = v32[3];
      if ( v35 > 0 )
      {
        v22 = 0;
        do
        {
          v23 = v20;
          v20 += 24;
          v24 = (const __m128i *)(*a3 + v22);
          *(__m128i *)(v20 - 24) = _mm_loadu_si128(v24);
          v22 += 24;
          *(_QWORD *)(v20 - 8) = v24[1].m128i_i64[0];
          v25 = -(((unsigned int)(v23 - v21) >> 3) + 10);
          *(_BYTE *)(v21 + v25) |= 1 << ((v23 - v21) & 7);
        }
        while ( 8 * (3LL * (unsigned int)(v35 - 1) + 3) != v22 );
      }
      return v37;
    }
  }
LABEL_24:
  if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
  {
    v29 = a1;
    goto LABEL_31;
  }
  return 0;
}
