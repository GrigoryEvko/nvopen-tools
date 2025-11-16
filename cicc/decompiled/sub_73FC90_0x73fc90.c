// Function: sub_73FC90
// Address: 0x73fc90
//
__m128i *__fastcall sub_73FC90(__int64 a1, __m128i *a2, unsigned int a3, _QWORD *a4)
{
  __m128i *v6; // rax
  _UNKNOWN *__ptr32 *v7; // r8
  __int64 v8; // rdx
  __m128i *v9; // r15
  __int8 v10; // al
  unsigned int v11; // r14d
  unsigned int v12; // r9d
  __int64 v14; // r13
  __int64 v15; // rax
  const __m128i *v16; // rdi
  __int64 v17; // r10
  _QWORD *v18; // rax
  __int64 v19; // rax
  const __m128i *v20; // rdi
  const __m128i *v21; // rax
  __int64 v22; // rax
  __int64 v23; // r10
  _QWORD *v24; // rax
  __int64 v25; // [rsp+8h] [rbp-68h]
  unsigned int v26; // [rsp+10h] [rbp-60h]
  unsigned int v27; // [rsp+10h] [rbp-60h]
  __int64 v28; // [rsp+10h] [rbp-60h]
  __int64 v29; // [rsp+10h] [rbp-60h]
  unsigned int v30; // [rsp+1Ch] [rbp-54h]
  _BOOL4 v31; // [rsp+20h] [rbp-50h]
  unsigned int v32; // [rsp+24h] [rbp-4Ch]
  bool v33; // [rsp+2Ah] [rbp-46h]
  bool v34; // [rsp+2Bh] [rbp-45h]
  unsigned int v36; // [rsp+2Ch] [rbp-44h]
  __m128i *v37; // [rsp+38h] [rbp-38h] BYREF

  v30 = a3 & 0x400;
  v33 = (a3 & 0x201) != 0;
  v34 = 1;
  v32 = a3 & 0x800;
  if ( (a3 & 0x800) == 0 )
    v34 = (*(_BYTE *)(a1 + 172) & 2) != 0;
  v6 = (__m128i *)sub_724DC0();
  v8 = a3;
  v37 = v6;
  v9 = v6;
  if ( a2 )
  {
    if ( (__m128i *)a1 != a2 )
    {
      sub_72A510((const __m128i *)a1, a2);
      v8 = a3;
    }
    v36 = 0;
    v9 = a2;
    v31 = (v8 & 0x100) == 0;
  }
  else
  {
    v26 = a3;
    v36 = a3 & 0x20;
    if ( (v8 & 0x20) != 0 )
    {
      sub_72A510((const __m128i *)a1, v6);
      v31 = 0;
      v8 = v26;
      v36 = 1;
    }
    else
    {
      v9 = (__m128i *)sub_724D50(*(_BYTE *)(a1 + 173));
      sub_72A510((const __m128i *)a1, v9);
      v31 = 1;
      v8 = v26;
    }
  }
  v10 = v9[10].m128i_i8[13];
  v11 = v8 & 0xFFFFFADF;
  switch ( v10 )
  {
    case 10:
      v14 = *(_QWORD *)(a1 + 176);
      v9[11].m128i_i64[0] = 0;
      for ( v9[11].m128i_i64[1] = 0; v14; v14 = *(_QWORD *)(v14 + 120) )
      {
        v15 = sub_73FC90(v14, 0, v11, a4);
        sub_72A690(v15, (__int64)v9, 0, 0);
      }
      break;
    case 11:
      v9[11].m128i_i64[0] = sub_73FC90(*(_QWORD *)(a1 + 176), 0, v11, a4);
      break;
    case 9:
      v9[11].m128i_i64[0] = (__int64)sub_73F780(*(_QWORD *)(a1 + 176), v11, a4);
      break;
    default:
      BYTE1(v8) &= 0xFAu;
      v12 = v8;
      if ( v10 == 6 )
      {
        if ( (unsigned __int8)(v9[11].m128i_i8[0] - 2) <= 1u )
        {
          v17 = *(_QWORD *)(a1 + 184);
          if ( v34
            || (*(_BYTE *)(v17 + 172) & 2) != 0
            || (*(_BYTE *)(v17 - 8) & 1) == 0 && ((v8 = (__int64)dword_4F07270, dword_4F07270[0] == unk_4F073B8) || v33) )
          {
            v18 = (_QWORD *)*a4;
            if ( !*a4 )
              goto LABEL_66;
            while ( v17 != v18[1] )
            {
              v18 = (_QWORD *)*v18;
              if ( !v18 )
                goto LABEL_66;
            }
            v19 = v18[2];
            if ( v19 )
            {
              v9[11].m128i_i64[1] = v19;
            }
            else
            {
LABEL_66:
              v28 = *(_QWORD *)(a1 + 184);
              v22 = sub_73FC90(v28, 0, v12, a4);
              v23 = v28;
              v9[11].m128i_i64[1] = v22;
              v8 = v22;
              v24 = (_QWORD *)qword_4F07AD8;
              if ( qword_4F07AD8 )
              {
                qword_4F07AD8 = *(_QWORD *)qword_4F07AD8;
              }
              else
              {
                v25 = v28;
                v29 = v8;
                v24 = (_QWORD *)sub_823970(24);
                v23 = v25;
                v8 = v29;
              }
              *v24 = *a4;
              *a4 = v24;
              v24[1] = v23;
              v24[2] = v8;
            }
          }
        }
        v16 = *(const __m128i **)(a1 + 200);
        if ( v16 )
          v9[12].m128i_i64[1] = sub_72A820(v16);
      }
      else if ( v10 == 12 )
      {
        switch ( v9[11].m128i_i8[0] )
        {
          case 0:
          case 2:
          case 3:
          case 0xD:
            goto LABEL_12;
          case 1:
            v9[11].m128i_i8[1] &= ~0x10u;
            v27 = v8;
            v21 = (const __m128i *)sub_72E9A0(a1);
            if ( !v21 )
              goto LABEL_12;
            if ( v34 || (v21[-1].m128i_i8[8] & 1) != 0 || (v8 = (__int64)dword_4F07270, dword_4F07270[0] != unk_4F073B8) )
              v21 = (const __m128i *)sub_73A9D0(v21, v27, (__int64)a4);
LABEL_59:
            v9[11].m128i_i64[1] = (__int64)v21;
            break;
          case 4:
          case 0xC:
            v21 = (const __m128i *)sub_73FC90(*(_QWORD *)(a1 + 184), 0, (unsigned int)v8, a4);
            goto LABEL_59;
          case 5:
          case 6:
          case 7:
          case 8:
          case 9:
          case 0xA:
            v9[11].m128i_i8[1] &= ~0x10u;
            v20 = *(const __m128i **)(a1 + 192);
            if ( v20 )
            {
              if ( v34
                || (v20[-1].m128i_i8[8] & 1) != 0
                || (v8 = (__int64)dword_4F07270, dword_4F07270[0] != unk_4F073B8) )
              {
                v20 = (const __m128i *)sub_73A9D0(v20, v12, (__int64)a4);
              }
              v9[12].m128i_i64[0] = (__int64)v20;
            }
            goto LABEL_12;
          case 0xB:
            v9[11].m128i_i64[1] = sub_73FC90(*(_QWORD *)(a1 + 184), 0, (unsigned int)v8, a4);
            v9[12].m128i_i64[0] = (__int64)sub_72F240(*(const __m128i **)(a1 + 192));
            goto LABEL_12;
          default:
            sub_721090();
        }
      }
      break;
  }
LABEL_12:
  if ( *(_QWORD *)(a1 + 144) )
  {
    if ( v33
      || v34
      || (v7 = (_UNKNOWN *__ptr32 *)v30, !v30) && v31 && (*(_BYTE *)(a1 - 8) & 1) != 0 && (v9[-1].m128i_i8[8] & 1) == 0 )
    {
      v9[9].m128i_i64[0] = 0;
    }
  }
  if ( v32 )
    v9[10].m128i_i8[12] |= 2u;
  if ( v36 )
  {
    v9 = (__m128i *)sub_73A460(v9, v32, v8, v36, v7);
  }
  else
  {
    if ( !v31 )
      goto LABEL_23;
    sub_73B910((__int64)v9);
  }
  if ( !v30 )
    v9[-1].m128i_i8[8] = *(_BYTE *)(a1 - 8) & 8 | v9[-1].m128i_i8[8] & 0xF7;
LABEL_23:
  sub_724E30((__int64)&v37);
  return v9;
}
