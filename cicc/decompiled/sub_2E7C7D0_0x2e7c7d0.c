// Function: sub_2E7C7D0
// Address: 0x2e7c7d0
//
__int64 __fastcall sub_2E7C7D0(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  unsigned __int64 *v3; // r8
  unsigned __int8 v4; // r11
  __m128i *v6; // rsi
  unsigned __int64 v7; // rax
  __int64 v8; // r9
  __int64 v9; // rbx
  __int64 v10; // r11
  unsigned int v11; // r13d
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rsi
  __int64 v15; // rdx
  __m128i v16; // rax
  __m128i v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rax
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int8 v27; // r11
  __m128i *v29; // rsi
  __int64 v30; // rax
  __int64 v31; // [rsp+0h] [rbp-70h]
  unsigned __int64 *v32; // [rsp+0h] [rbp-70h]
  unsigned __int64 *v33; // [rsp+8h] [rbp-68h]
  __int64 v34; // [rsp+8h] [rbp-68h]
  __int64 v35; // [rsp+10h] [rbp-60h]
  unsigned __int64 *v36; // [rsp+10h] [rbp-60h]
  unsigned __int64 *v37; // [rsp+10h] [rbp-60h]
  char v38; // [rsp+1Eh] [rbp-52h]
  unsigned __int64 v40; // [rsp+20h] [rbp-50h]
  __int64 v41; // [rsp+20h] [rbp-50h]
  __int64 v42; // [rsp+20h] [rbp-50h]
  __int64 v43; // [rsp+20h] [rbp-50h]
  __int64 v44; // [rsp+20h] [rbp-50h]
  unsigned __int64 *v45; // [rsp+28h] [rbp-48h]
  __m128i v46; // [rsp+30h] [rbp-40h] BYREF

  v3 = (unsigned __int64 *)a1;
  v4 = a3;
  if ( a3 > *(_BYTE *)a1 )
    *(_BYTE *)a1 = a3;
  v6 = *(__m128i **)(a1 + 16);
  v7 = *(_QWORD *)(a1 + 8);
  v8 = (__int64)((__int64)v6->m128i_i64 - v7) >> 4;
  if ( !(_DWORD)v8 )
  {
LABEL_28:
    v46.m128i_i16[4] = v4;
    v46.m128i_i64[0] = a2;
    if ( (__m128i *)v3[3] == v6 )
    {
      v45 = v3;
      sub_2E7C650(v3 + 1, v6, &v46);
      v3 = v45;
      v29 = (__m128i *)v45[2];
    }
    else
    {
      if ( v6 )
      {
        *v6 = _mm_loadu_si128(&v46);
        v6 = (__m128i *)v3[2];
      }
      v29 = v6 + 1;
      v3[2] = (unsigned __int64)v29;
    }
    return (unsigned int)((__int64)((__int64)v29->m128i_i64 - v3[1]) >> 4) - 1;
  }
  v9 = 0;
  v10 = (unsigned int)v8;
  while ( 1 )
  {
    v11 = v9;
    v12 = 16 * v9 + v7;
    if ( *(_BYTE *)(v12 + 9) )
      goto LABEL_5;
    v13 = *(_QWORD *)v12;
    if ( *(_QWORD *)v12 == a2 )
      break;
    v14 = *(_QWORD *)(v13 + 8);
    v15 = *(_QWORD *)(a2 + 8);
    if ( v14 != v15
      && (unsigned __int8)(*(_BYTE *)(v14 + 8) - 15) > 1u
      && (unsigned __int8)(*(_BYTE *)(v15 + 8) - 15) > 1u )
    {
      v31 = v10;
      v33 = v3;
      v35 = v3[8];
      v16.m128i_i64[0] = (unsigned __int64)(sub_9208B0(v35, v14) + 7) >> 3;
      v46 = v16;
      v40 = sub_CA1930(&v46);
      v17.m128i_i64[0] = (unsigned __int64)(sub_9208B0(v35, *(_QWORD *)(a2 + 8)) + 7) >> 3;
      v46 = v17;
      v18 = sub_CA1930(&v46);
      v3 = v33;
      v10 = v31;
      if ( v40 == v18 && v40 <= 0x80 )
      {
        v32 = v33;
        v34 = v10;
        v38 = sub_AD6C40(v13);
        v19 = (_QWORD *)sub_BD5C60(v13);
        v20 = sub_BCCE00(v19, 8 * (int)v40);
        v10 = v34;
        v3 = v32;
        v21 = v20;
        v22 = *(_QWORD *)(v13 + 8);
        v23 = v35;
        if ( *(_BYTE *)(v22 + 8) == 14 )
        {
          v43 = v21;
          v30 = sub_96F480(0x2Fu, v13, v21, v35);
          v21 = v43;
          v23 = v35;
          v10 = v34;
          v3 = v32;
          v13 = v30;
        }
        else if ( v21 != v22 )
        {
          v41 = v21;
          v24 = sub_96F480(0x31u, v13, v21, v35);
          v3 = v32;
          v10 = v34;
          v23 = v35;
          v21 = v41;
          v13 = v24;
        }
        v25 = *(_QWORD *)(a2 + 8);
        if ( *(_BYTE *)(v25 + 8) == 14 )
        {
          v37 = v3;
          v44 = v10;
          v26 = sub_96F480(0x2Fu, a2, v21, v23);
          v10 = v44;
          v3 = v37;
        }
        else
        {
          v26 = a2;
          if ( v21 != v25 )
          {
            v36 = v3;
            v42 = v10;
            v26 = sub_96F480(0x31u, a2, v21, v23);
            v3 = v36;
            v10 = v42;
          }
        }
        if ( v13 == v26 && v38 != 1 )
        {
          v27 = a3;
          v12 = 16 * v9 + v3[1];
          goto LABEL_24;
        }
      }
    }
LABEL_5:
    if ( v10 == ++v9 )
    {
      v4 = a3;
      v6 = (__m128i *)v3[2];
      goto LABEL_28;
    }
    v7 = v3[1];
  }
  v27 = a3;
LABEL_24:
  if ( v27 > *(_BYTE *)(v12 + 8) )
    *(_BYTE *)(v12 + 8) = v27;
  return v11;
}
