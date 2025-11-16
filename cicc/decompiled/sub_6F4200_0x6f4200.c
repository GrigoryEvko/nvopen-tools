// Function: sub_6F4200
// Address: 0x6f4200
//
__int64 __fastcall sub_6F4200(__m128i *a1, __int64 a2, int a3, int a4)
{
  __int64 v7; // rcx
  int v8; // ebx
  __int64 v9; // r8
  __int64 v10; // r9
  __int8 v11; // al
  __int64 v12; // rsi
  __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 i; // rdx
  __int64 v19; // r11
  unsigned int v20; // r8d
  __int64 v21; // rdi
  __int64 result; // rax
  unsigned int v23; // eax
  int v24; // eax
  __int64 v25; // rax
  __int64 v26; // r11
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  int v31; // eax
  __int64 v32; // rdi
  int v33; // eax
  __int64 v34; // r9
  int v35; // eax
  __int64 *v36; // [rsp+0h] [rbp-1C0h]
  unsigned int v37; // [rsp+0h] [rbp-1C0h]
  __int64 v38; // [rsp+8h] [rbp-1B8h]
  __int64 v39; // [rsp+8h] [rbp-1B8h]
  __int64 v40; // [rsp+8h] [rbp-1B8h]
  __int64 v41; // [rsp+8h] [rbp-1B8h]
  unsigned int v42; // [rsp+8h] [rbp-1B8h]
  __int64 v43; // [rsp+8h] [rbp-1B8h]
  _BOOL4 v44; // [rsp+14h] [rbp-1ACh]
  unsigned int v45; // [rsp+18h] [rbp-1A8h]
  __int64 v47; // [rsp+28h] [rbp-198h] BYREF
  _OWORD v48[4]; // [rsp+30h] [rbp-190h] BYREF
  _OWORD v49[5]; // [rsp+70h] [rbp-150h] BYREF
  __m128i v50; // [rsp+C0h] [rbp-100h]
  __m128i v51; // [rsp+D0h] [rbp-F0h]
  __m128i v52; // [rsp+E0h] [rbp-E0h]
  __m128i v53; // [rsp+F0h] [rbp-D0h]
  __m128i v54; // [rsp+100h] [rbp-C0h]
  __m128i v55; // [rsp+110h] [rbp-B0h]
  __m128i v56; // [rsp+120h] [rbp-A0h]
  __m128i v57; // [rsp+130h] [rbp-90h]
  __m128i v58; // [rsp+140h] [rbp-80h]
  __m128i v59; // [rsp+150h] [rbp-70h]
  __m128i v60; // [rsp+160h] [rbp-60h]
  __m128i v61; // [rsp+170h] [rbp-50h]
  __m128i v62; // [rsp+180h] [rbp-40h]

  v8 = sub_8D32E0(a2);
  v48[0] = _mm_loadu_si128(a1);
  v48[1] = _mm_loadu_si128(a1 + 1);
  v11 = a1[1].m128i_i8[0];
  v48[2] = _mm_loadu_si128(a1 + 2);
  v48[3] = _mm_loadu_si128(a1 + 3);
  v49[0] = _mm_loadu_si128(a1 + 4);
  v49[1] = _mm_loadu_si128(a1 + 5);
  v49[2] = _mm_loadu_si128(a1 + 6);
  v49[3] = _mm_loadu_si128(a1 + 7);
  v49[4] = _mm_loadu_si128(a1 + 8);
  if ( v11 == 2 )
  {
    v50 = _mm_loadu_si128(a1 + 9);
    v51 = _mm_loadu_si128(a1 + 10);
    v52 = _mm_loadu_si128(a1 + 11);
    v53 = _mm_loadu_si128(a1 + 12);
    v54 = _mm_loadu_si128(a1 + 13);
    v55 = _mm_loadu_si128(a1 + 14);
    v56 = _mm_loadu_si128(a1 + 15);
    v57 = _mm_loadu_si128(a1 + 16);
    v58 = _mm_loadu_si128(a1 + 17);
    v59 = _mm_loadu_si128(a1 + 18);
    v60 = _mm_loadu_si128(a1 + 19);
    v61 = _mm_loadu_si128(a1 + 20);
    v62 = _mm_loadu_si128(a1 + 21);
    goto LABEL_4;
  }
  if ( v11 != 5 && v11 != 1 )
  {
LABEL_4:
    if ( !v8 )
      goto LABEL_5;
    goto LABEL_28;
  }
  v50.m128i_i64[0] = a1[9].m128i_i64[0];
  if ( !v8 )
  {
LABEL_5:
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) > 3u )
    {
      if ( (unsigned int)sub_8D3A70(a2) || (unsigned int)sub_8D3D40(a2) || (unsigned int)sub_8D3A70(a1->m128i_i64[0]) )
      {
        v45 = 0;
        v12 = 0;
        v14 = 0;
        v13 = a2;
      }
      else
      {
        v13 = a2;
        v12 = 0;
        v45 = 0;
        v14 = (unsigned int)sub_8D3D40(a1->m128i_i64[0]) == 0;
      }
    }
    else
    {
      v45 = 0;
      v12 = 0;
      v13 = a2;
      v14 = 1;
    }
    goto LABEL_7;
  }
LABEL_28:
  v13 = sub_8D46C0(a2);
  v23 = sub_8D3110(a2);
  v12 = 1;
  v14 = v23;
  if ( v23 )
  {
    v14 = (unsigned int)sub_8D3D40(v13) == 0;
    v12 = 0;
  }
  v45 = v14;
  if ( *(_BYTE *)(a2 + 140) == 12 )
  {
    v7 = *(unsigned __int8 *)(a2 + 184);
    if ( (unsigned __int8)v7 <= 0xCu && ((0x18C2uLL >> v7) & 1) != 0 )
      v13 = a2;
  }
LABEL_7:
  sub_6F3DD0((__int64)a1, v12, v14, v7, v9, v10);
  v16 = a1[1].m128i_u8[0];
  if ( !(_BYTE)v16 )
    goto LABEL_25;
  v17 = a1->m128i_i64[0];
  for ( i = *(unsigned __int8 *)(a1->m128i_i64[0] + 140); (_BYTE)i == 12; i = *(unsigned __int8 *)(v17 + 140) )
    v17 = *(_QWORD *)(v17 + 160);
  if ( !(_BYTE)i )
  {
LABEL_25:
    sub_6E6870((__int64)a1);
    goto LABEL_26;
  }
  if ( (_BYTE)v16 != 6 )
  {
    v44 = (_BYTE)v16 == 2 && !v8 && (a1[1].m128i_i8[1] == 2 || sub_6ED0A0((__int64)a1)) && sub_8D3A70(v13) == 0;
    if ( a4 )
    {
      v12 = v13;
      if ( (unsigned int)sub_8DAAE0(a1->m128i_i64[0], v13) )
        goto LABEL_26;
    }
    if ( v45 )
    {
      v12 = 1;
      sub_6F3DD0((__int64)a1, 1, 0, v16, v15, v45);
    }
    if ( a3 == 2 )
    {
      v19 = sub_6F7150(a1, v12, i);
    }
    else
    {
      v19 = sub_6F6F40(a1, 0);
      if ( (unsigned int)(a3 - 4) <= 1 )
      {
        if ( a3 == 7 )
        {
LABEL_21:
          v20 = 18;
          if ( v8 )
          {
            v21 = 19;
            goto LABEL_59;
          }
LABEL_42:
          if ( (*(_BYTE *)(v19 + 25) & 3) != 0 )
          {
            v36 = (__int64 *)v19;
            v42 = v20;
            v31 = sub_8DC060(v13);
            v20 = v42;
            v19 = (__int64)v36;
            if ( !v31 )
            {
              v32 = *v36;
              v37 = v42;
              v43 = v19;
              v33 = sub_8DC060(v32);
              v19 = v43;
              v20 = v37;
              if ( !v33 )
              {
                sub_6ED3D0(v43, 0, 0, 0, v37, v34);
                v20 = v37;
                v19 = v43;
              }
            }
          }
          v21 = v20;
          v39 = v19;
          v25 = sub_73DBF0(v20, v13, v19);
          v26 = v39;
          v27 = v25;
          if ( !a4 )
            goto LABEL_45;
LABEL_44:
          v28 = *(_QWORD *)(v26 + 28);
          *(_BYTE *)(v27 + 27) |= 2u;
          *(_QWORD *)(v27 + 28) = v28;
          if ( !v8 )
          {
LABEL_45:
            switch ( a3 )
            {
              case 0:
              case 1:
              case 2:
              case 7:
                break;
              case 3:
                *(_BYTE *)(v27 + 25) |= 0x40u;
                break;
              case 4:
                *(_BYTE *)(v27 + 58) |= 8u;
                break;
              case 5:
                *(_BYTE *)(v27 + 58) |= 2u;
                break;
              default:
                sub_721090(v21);
            }
            sub_6E7170((__int64 *)v27, (__int64)a1);
            if ( v45 )
              sub_6ED1A0((__int64)a1);
            if ( v44 )
              sub_6F4B70(a1);
            goto LABEL_26;
          }
LABEL_60:
          v21 = v27;
          sub_6E84C0(v27, a2);
          *(_BYTE *)(v27 + 25) |= 1u;
          goto LABEL_45;
        }
        goto LABEL_40;
      }
      if ( a3 == 7 )
        goto LABEL_21;
    }
    v38 = v19;
    v24 = sub_8D3A70(v13);
    v19 = v38;
    if ( v24 )
    {
      if ( !v8 )
      {
        if ( !dword_4F077BC
          || (_DWORD)qword_4F077B4
          || (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0
          || (v35 = sub_8D23B0(v13), v19 = v38, !v35) )
        {
          v21 = v13;
          v41 = v19;
          v27 = sub_6ECAE0(v13, 0, a4 == 0, 0, 5u, (__int64 *)((char *)v49 + 4), &v47);
          v30 = v47;
          *(_QWORD *)(v47 + 56) = 0;
          *(_QWORD *)(v30 + 64) = v41;
          goto LABEL_45;
        }
        goto LABEL_41;
      }
LABEL_58:
      v21 = 7;
LABEL_59:
      v40 = v19;
      v29 = sub_73DBF0(v21, v13, v19);
      v26 = v40;
      v27 = v29;
      if ( !a4 )
        goto LABEL_60;
      goto LABEL_44;
    }
LABEL_40:
    if ( !v8 )
    {
LABEL_41:
      v20 = 5;
      goto LABEL_42;
    }
    goto LABEL_58;
  }
  if ( (unsigned int)sub_6E5430() )
    sub_6851F0(0x14u, *(_QWORD *)(*(_QWORD *)a1[8].m128i_i64[1] + 8LL));
  sub_6E6260(a1);
LABEL_26:
  result = sub_6E4F10((__int64)a1, (__int64)v48, a4, 1);
  a1[1].m128i_i8[2] &= 0xD7u;
  return result;
}
