// Function: sub_8A3C00
// Address: 0x8a3c00
//
__m128i *__fastcall sub_8A3C00(__int64 a1, __int64 a2, int a3, __int64 *a4)
{
  __int64 v5; // rax
  int v6; // ecx
  __int64 v7; // rsi
  int v8; // edx
  const __m128i *v9; // rbx
  char v10; // dl
  __int64 v11; // r13
  __m128i *v12; // r12
  int v13; // r14d
  _BOOL4 v14; // r15d
  char v15; // al
  __m128i *v16; // r10
  __m128i *v17; // rax
  _QWORD *m128i_i64; // r10
  __m128i *v19; // rax
  __m128i *v20; // r8
  __int8 v21; // al
  char v22; // di
  __int8 v23; // al
  __m128i *v24; // rax
  __int8 v25; // al
  __m128i *v26; // rax
  char v27; // al
  _BYTE *v28; // rax
  int v29; // edi
  __m128i *v31; // rax
  __m128i *v32; // rax
  __m128i *v33; // rax
  __int64 v34; // rdx
  _DWORD *v35; // r8
  int v36; // eax
  _QWORD *v37; // r10
  __int64 v38; // rax
  __int64 **v39; // rax
  int v40; // eax
  bool v41; // dl
  _QWORD *v44; // [rsp+18h] [rbp-C8h]
  _QWORD *v45; // [rsp+18h] [rbp-C8h]
  _QWORD *v46; // [rsp+18h] [rbp-C8h]
  __int64 *v48; // [rsp+28h] [rbp-B8h]
  char v49; // [rsp+28h] [rbp-B8h]
  __int64 v50; // [rsp+28h] [rbp-B8h]
  __int64 *v51; // [rsp+28h] [rbp-B8h]
  __int64 *v52; // [rsp+28h] [rbp-B8h]
  bool v53; // [rsp+33h] [rbp-ADh]
  int v54; // [rsp+34h] [rbp-ACh]
  __m128i *v55; // [rsp+38h] [rbp-A8h]
  int v56; // [rsp+44h] [rbp-9Ch] BYREF
  __int64 **v57; // [rsp+48h] [rbp-98h] BYREF
  __m128i v58[9]; // [rsp+50h] [rbp-90h] BYREF

  if ( a2 )
  {
    if ( !a1 )
    {
      v5 = a2;
      v6 = 0;
      goto LABEL_98;
    }
    v5 = a2;
    v6 = 0;
    v7 = a1;
    while ( 1 )
    {
      while ( 1 )
      {
        v8 = *(unsigned __int8 *)(v5 + 8);
        if ( (_BYTE)v8 != 3 )
          break;
        v5 = *(_QWORD *)v5;
        v6 = 1;
        if ( !v5 )
          goto LABEL_6;
      }
      v22 = *(_BYTE *)(*(_QWORD *)(v7 + 8) + 80LL);
      if ( v22 == 3 )
        break;
      if ( v22 == 2 )
      {
        v29 = 1;
LABEL_66:
        if ( v8 != v29 )
          return 0;
        goto LABEL_36;
      }
      if ( (_BYTE)v8 != 2 && (*(char *)(v5 + 24) >= 0 || v22 != 19) )
        return 0;
LABEL_36:
      if ( (*(_BYTE *)(v7 + 56) & 0x10) == 0 )
        v7 = *(_QWORD *)v7;
      v5 = *(_QWORD *)v5;
      if ( !v7 )
      {
        if ( v5 )
        {
LABEL_98:
          v41 = 0;
          if ( *(_BYTE *)(v5 + 8) != 3 )
            v41 = v6 == 0;
          if ( !a3 || v41 || !a1 )
            return 0;
          v53 = 1;
          v9 = (const __m128i *)a2;
          v10 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 80LL);
        }
        else
        {
LABEL_6:
          v9 = (const __m128i *)a2;
          v53 = a3 != 0;
          v10 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 80LL);
        }
LABEL_7:
        v11 = a1;
        v12 = 0;
        v13 = 0;
        v14 = 0;
        v55 = 0;
        v54 = 0;
        while ( 2 )
        {
          v15 = *(_BYTE *)(v11 + 56) & 0x10;
          if ( v9 && v9->m128i_i8[8] == 3 )
            goto LABEL_21;
LABEL_10:
          if ( v15 )
          {
            v16 = v12;
            if ( !v13 )
            {
              v49 = v10;
              v26 = (__m128i *)sub_725090(3u);
              v10 = v49;
              v16 = v26;
              if ( v55 )
                v12->m128i_i64[0] = (__int64)v26;
              else
                v55 = v26;
              v27 = *(_BYTE *)(v11 + 56);
              if ( (v27 & 0x40) != 0 )
              {
                v14 = 1;
                v13 = 1;
                v54 = *(_DWORD *)(v11 + 60);
              }
              else
              {
                if ( (v27 & 0x10) == 0 )
                {
                  v12 = v16;
                  v13 = 1;
                  goto LABEL_11;
                }
                v13 = 1;
              }
            }
          }
          else
          {
LABEL_11:
            v16 = v12;
            if ( !v14 )
            {
              if ( !v9 )
              {
                if ( v10 == 3 )
                {
                  v33 = (__m128i *)sub_725090(0);
                  m128i_i64 = v12->m128i_i64;
                  v12 = v33;
                  if ( !m128i_i64 )
                    goto LABEL_79;
LABEL_17:
                  *m128i_i64 = v12;
                  if ( v9 )
                    goto LABEL_18;
                }
                else
                {
                  if ( v10 == 2 )
                    v17 = (__m128i *)sub_725090(1u);
                  else
                    v17 = (__m128i *)sub_725090(2u);
                  m128i_i64 = v12->m128i_i64;
                  v12 = v17;
                  if ( m128i_i64 )
                    goto LABEL_17;
LABEL_79:
                  v55 = v12;
                  v9 = 0;
                }
LABEL_25:
                v11 = *(_QWORD *)v11;
                if ( !v11 )
                  return v55;
                v10 = *(_BYTE *)(*(_QWORD *)(v11 + 8) + 80LL);
                if ( !v9 && v53 )
                  return v55;
                if ( v14 )
                  v14 = *(_DWORD *)(v11 + 60) == v54;
                v13 = 0;
                continue;
              }
LABEL_74:
              v23 = v9->m128i_i8[8];
              v16 = v12;
              v14 = 0;
              goto LABEL_45;
            }
          }
          break;
        }
        if ( !v9 )
        {
          v12 = v16;
          goto LABEL_25;
        }
        v23 = v9->m128i_i8[8];
LABEL_45:
        if ( v23 == 3 )
        {
          v12 = v16;
          while ( 1 )
          {
            v19 = (__m128i *)sub_725090(3u);
            m128i_i64 = v12->m128i_i64;
            v20 = v19;
            v21 = v9->m128i_i8[8];
            v12 = v20;
LABEL_50:
            if ( v21 != 3 )
              break;
LABEL_55:
            if ( m128i_i64 )
              goto LABEL_17;
            v55 = v12;
LABEL_18:
            v9 = (const __m128i *)v9->m128i_i64[0];
            if ( v14 | v13 ^ 1 || !v9 )
              goto LABEL_25;
            v14 = 0;
            v13 = 1;
            v10 = *(_BYTE *)(*(_QWORD *)(v11 + 8) + 80LL);
            v15 = *(_BYTE *)(v11 + 56) & 0x10;
            if ( v9->m128i_i8[8] != 3 )
              goto LABEL_10;
LABEL_21:
            if ( v15 )
            {
              v13 = 1;
            }
            else
            {
              v13 = 1;
              if ( !v14 )
                goto LABEL_74;
              v13 = v14;
            }
          }
        }
        else if ( v10 == 3 )
        {
          v51 = (__int64 *)v16;
          v31 = (__m128i *)sub_725090(0);
          m128i_i64 = v51;
          v12 = v31;
          if ( v9->m128i_i8[8] )
            goto LABEL_70;
        }
        else if ( v10 == 2 )
        {
          v52 = (__int64 *)v16;
          v32 = (__m128i *)sub_725090(1u);
          m128i_i64 = v52;
          v12 = v32;
          if ( v9->m128i_i8[8] != 1 )
            goto LABEL_70;
        }
        else
        {
          v48 = (__int64 *)v16;
          v24 = (__m128i *)sub_725090(2u);
          m128i_i64 = v48;
          v12 = v24;
          v21 = v9->m128i_i8[8];
          if ( v21 != 2 )
          {
            if ( v9[1].m128i_i8[8] >= 0 )
              goto LABEL_70;
            goto LABEL_50;
          }
        }
        v12[1].m128i_i8[8] = v12[1].m128i_i8[8] & 0xF5 | v9[1].m128i_i8[8] & 2 | (8 * (v14 | v13)) & 0xA;
        v25 = v12->m128i_i8[8];
        if ( !v25 )
        {
LABEL_64:
          v12[2].m128i_i64[0] = v9[2].m128i_i64[0];
          goto LABEL_55;
        }
        if ( v25 == 2 )
        {
          if ( v9[1].m128i_i8[8] < 0 )
          {
            v38 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v9[2].m128i_i64[0] + 96LL) + 72LL);
            if ( !v38 )
              goto LABEL_70;
            v12[2].m128i_i64[0] = *(_QWORD *)(*(_QWORD *)(v38 + 88) + 104LL);
          }
          else
          {
            v12[2] = _mm_loadu_si128(v9 + 2);
          }
          goto LABEL_55;
        }
        v44 = m128i_i64;
        v57 = 0;
        v28 = sub_724D80(0);
        m128i_i64 = v44;
        v50 = (__int64)v28;
        if ( (*(_BYTE *)(v11 + 57) & 8) != 0 )
        {
          if ( v53 )
            goto LABEL_64;
          v34 = v9[3].m128i_i64[0];
LABEL_84:
          v35 = (_DWORD *)(v34 + 76);
          v45 = m128i_i64;
          if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) & 0x10) != 0 )
            v35 = 0;
          if ( !(unsigned int)sub_696F90(
                                *(_QWORD *)(*(_QWORD *)(v11 + 64) + 128LL),
                                0,
                                v34,
                                (__int64 *)&v57,
                                v35,
                                a2,
                                a1)
            || (v36 = sub_695E90(v9[3].m128i_i64[0], (__int64)v57), v37 = v45, !v36) )
          {
LABEL_70:
            if ( v55 )
              sub_725130(v55->m128i_i64);
            return 0;
          }
        }
        else
        {
          v56 = 0;
          sub_892150(v58);
          v57 = *(__int64 ***)(*(_QWORD *)(v11 + 64) + 128LL);
          v39 = sub_8A2270((__int64)v57, v55, a1, a4, 0, &v56, v58);
          v57 = v39;
          if ( v56 )
            goto LABEL_70;
          m128i_i64 = v44;
          if ( v53 )
            goto LABEL_64;
          v34 = v9[3].m128i_i64[0];
          if ( (*(_BYTE *)(v11 + 57) & 8) != 0 )
            goto LABEL_84;
          v40 = sub_695E90(v9[3].m128i_i64[0], (__int64)v39);
          v37 = v44;
          if ( !v40 )
            goto LABEL_70;
        }
        v46 = v37;
        sub_696090(v9[3].m128i_i64[0], (__int64)v57, v50);
        m128i_i64 = v46;
        if ( !*(_BYTE *)(v50 + 173) )
          goto LABEL_70;
        v12[3].m128i_i64[0] = 0;
        v12[2].m128i_i64[0] = v50;
        goto LABEL_55;
      }
      if ( !v5 )
        goto LABEL_6;
    }
    v29 = 0;
    goto LABEL_66;
  }
  if ( a1 )
  {
    v53 = a3 != 0;
    v10 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 80LL);
    if ( !a3 )
    {
      v9 = 0;
      goto LABEL_7;
    }
  }
  return 0;
}
