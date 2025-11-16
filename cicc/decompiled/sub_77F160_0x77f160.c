// Function: sub_77F160
// Address: 0x77f160
//
__int64 __fastcall sub_77F160(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5)
{
  __int64 v6; // r14
  unsigned int v7; // eax
  _WORD *v8; // r10
  __int64 v10; // r15
  char v11; // dl
  unsigned int v12; // ecx
  __int64 result; // rax
  int v14; // edx
  int v15; // ecx
  _QWORD *v16; // rax
  _BYTE *v17; // r15
  __m128i *v18; // r12
  __int64 v19; // rax
  __m128i v20; // xmm1
  __m128i v21; // xmm2
  __m128i v22; // xmm3
  __int64 v23; // rax
  size_t v24; // rax
  __int64 v25; // rax
  unsigned int v26; // ecx
  __int64 v27; // rdx
  unsigned int v28; // eax
  _WORD *v29; // [rsp+8h] [rbp-108h]
  __int64 v30; // [rsp+8h] [rbp-108h]
  _WORD *v31; // [rsp+8h] [rbp-108h]
  unsigned int v32; // [rsp+18h] [rbp-F8h] BYREF
  unsigned int v33; // [rsp+1Ch] [rbp-F4h] BYREF
  unsigned int v34; // [rsp+20h] [rbp-F0h] BYREF
  int v35; // [rsp+24h] [rbp-ECh] BYREF
  __int64 v36; // [rsp+28h] [rbp-E8h] BYREF
  _QWORD v37[2]; // [rsp+30h] [rbp-E0h] BYREF
  __m128i v38; // [rsp+40h] [rbp-D0h]
  __m128i v39; // [rsp+50h] [rbp-C0h]
  __m128i v40; // [rsp+60h] [rbp-B0h]
  char s[160]; // [rsp+70h] [rbp-A0h] BYREF

  v6 = *a4;
  v32 = 1;
  v7 = *(unsigned __int8 *)(v6 + 8);
  if ( (v7 & 0x21) != 0 )
  {
    v32 = 0;
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xA8Du, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return v32;
    }
    return 0;
  }
  if ( !*(_QWORD *)v6 )
  {
    v32 = 0;
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xACEu, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return v32;
    }
    return 0;
  }
  v8 = (_WORD *)a4[1];
  v10 = *(_QWORD *)(*(_QWORD *)(a2 + 240) + 32LL);
  v11 = *(_BYTE *)(v10 + 140);
  if ( (v7 & 1) == 0 )
  {
    v26 = 16;
    if ( (unsigned __int8)(v11 - 2) > 1u )
    {
      v31 = v8;
      v28 = sub_7764B0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 240) + 32LL), &v32);
      v8 = v31;
      v26 = v28;
      result = v32;
      if ( !v32 )
        return result;
      v7 = *(unsigned __int8 *)(v6 + 8);
    }
    if ( (v7 & 8) != 0 )
    {
      v34 = *(_DWORD *)(v6 + 8) >> 8;
      v27 = *(_QWORD *)(v6 + 16);
      if ( (v7 & 4) != 0 )
        v27 = *(_QWORD *)(v27 + 24);
      if ( v26 )
        v33 = ((unsigned int)*(_QWORD *)v6 - (unsigned int)v27) / v26;
      else
        v33 = 0;
    }
    else
    {
      v34 = 1;
      v33 = (v7 >> 1) & 1;
    }
LABEL_8:
    sub_620E00(v8, 0, &v36, &v35);
    if ( v35 )
    {
      v14 = v36;
      v15 = v34 - v33;
    }
    else
    {
      v14 = v36;
      v15 = v34 - v33;
      if ( v34 - v33 >= v36 )
      {
        if ( v36 )
        {
          v16 = sub_7259C0(8);
          v16[20] = v10;
          v30 = (__int64)v16;
          v16[22] = v36;
          v17 = sub_724D80(10);
          result = sub_77D750(a1, *(__m128i **)v6, *(_QWORD *)(v6 + 24), v30, (__int64)v17);
          if ( (_DWORD)result )
          {
            v18 = sub_735FB0(v30, 2, -1);
            sub_72FC40((__int64)v18, *(_QWORD *)(unk_4D03FF0 + 8LL));
            v18[11].m128i_i64[1] = (__int64)v17;
            v18[10].m128i_i8[12] |= 8u;
            v19 = qword_4F08078;
            v18[11].m128i_i8[1] = 1;
            qword_4F08078 = v19 + 1;
            sprintf(s, "__ce_array_%ld", v19 + 1);
            v20 = _mm_loadu_si128(&xmmword_4F06660[1]);
            v21 = _mm_loadu_si128(&xmmword_4F06660[2]);
            v22 = _mm_loadu_si128(&xmmword_4F06660[3]);
            v23 = *(_QWORD *)(a3 + 28);
            v37[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
            v38 = v20;
            v39 = v21;
            v40 = v22;
            v37[1] = v23;
            v24 = strlen(s);
            sub_878540(s, v24);
            v25 = sub_87EF90(7, v37);
            *(_QWORD *)(v25 + 88) = v18;
            sub_877D80(v18, v25);
            *(_BYTE *)a5 = 7;
            result = v32;
            *(_QWORD *)(a5 + 8) = v18;
            *(_DWORD *)(a5 + 16) = 0;
          }
          return result;
        }
        v32 = 0;
        if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
        {
          sub_6855B0(0xD25u, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
          sub_770D30(a1);
          return v32;
        }
        return 0;
      }
    }
    v32 = 0;
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_67E4F0(0xD26u, (_DWORD *)(a3 + 28), v14, v15, (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return v32;
    }
    return 0;
  }
  v12 = 1;
  if ( v11 != 1 )
    v12 = *(_DWORD *)(v10 + 128);
  v29 = v8;
  sub_771560(a1, *(_QWORD *)(v6 + 16), v10, v12, &v34, &v33, &v32);
  result = v32;
  if ( v32 )
  {
    v8 = v29;
    goto LABEL_8;
  }
  return result;
}
