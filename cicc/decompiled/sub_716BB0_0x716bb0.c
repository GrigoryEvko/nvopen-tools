// Function: sub_716BB0
// Address: 0x716bb0
//
__int64 __fastcall sub_716BB0(_QWORD *a1, __m128i *a2, __int64 a3, __int64 a4, unsigned int *a5, __int64 a6)
{
  unsigned int *v8; // rbx
  __int64 v9; // rdi
  __int64 v10; // r8
  __int64 v11; // rsi
  char v12; // al
  unsigned int v13; // r14d
  _QWORD *v15; // r15
  unsigned __int8 v16; // r14
  __int64 v17; // r9
  const __m128i *v18; // rsi
  __int64 v19; // r14
  char v20; // al
  __int64 v21; // r12
  __int64 v22; // rax
  int v23; // eax
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // r14
  __int64 v30; // r14
  __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // r12
  char v35; // al
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // r12
  __int64 v39; // rax
  __int64 v40; // r14
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // rax
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // [rsp+8h] [rbp-68h]
  unsigned int v47; // [rsp+14h] [rbp-5Ch]
  unsigned int v48; // [rsp+14h] [rbp-5Ch]
  char v49; // [rsp+14h] [rbp-5Ch]
  unsigned int v50; // [rsp+18h] [rbp-58h]
  unsigned int v51; // [rsp+18h] [rbp-58h]
  unsigned int v52; // [rsp+28h] [rbp-48h] BYREF
  int v53; // [rsp+2Ch] [rbp-44h] BYREF
  const __m128i *v54; // [rsp+30h] [rbp-40h] BYREF
  __int64 v55[7]; // [rsp+38h] [rbp-38h] BYREF

  v8 = a5;
  if ( !a5 )
    v8 = &v52;
  *v8 = 0;
  v9 = word_4D04898;
  v10 = (unsigned int)dword_4F04C44;
  v11 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  while ( 2 )
  {
    v12 = *((_BYTE *)a1 + 24);
    if ( word_4D04898
      && (*((_BYTE *)a1 + 25) & 3) != 0
      && (v12 == 1 || v12 == 23)
      && dword_4F04C44 == -1
      && (*(_BYTE *)(v11 + 6) & 6) == 0
      && *(_BYTE *)(v11 + 4) != 12 )
    {
      if ( dword_4D03F94 )
        return 0;
      v11 = (__int64)a2;
      v9 = (__int64)a1;
      v13 = sub_70D5B0((__int64)a1, (__int64)a2);
LABEL_10:
      if ( v8 != &v52 || !v13 )
        return v13;
LABEL_37:
      if ( v52 && a2[10].m128i_i8[13] != 12 )
      {
        v55[0] = sub_724DC0(v9, v11, v52, a4, v10, a6);
        sub_72A510(a2, v55[0]);
        sub_70FDD0(v55[0], (__int64)a2, a2[8].m128i_i64[0], 0);
        sub_724E30(v55);
      }
      return v13;
    }
    switch ( v12 )
    {
      case 0:
        v9 = (__int64)a2;
        sub_72C970(a2);
        goto LABEL_18;
      case 1:
        v15 = (_QWORD *)a1[9];
        v16 = *((_BYTE *)a1 + 56);
        v47 = a4;
        v50 = a3;
        v46 = v15[2];
        v54 = (const __m128i *)sub_724DC0(word_4D04898, v11, a3, a4, (unsigned int)dword_4F04C44, a6);
        v18 = v54;
        if ( v16 > 0x74u )
          goto LABEL_27;
        if ( v16 <= 0x5Bu )
        {
          if ( v16 > 8u )
          {
            if ( v16 != 14 || !(unsigned int)sub_716BB0(v15, v54, v50, v47, v8) )
              goto LABEL_27;
            if ( (unsigned int)sub_8DBE70(*a1) || (v13 = *v8) != 0 )
            {
              v11 = (__int64)a2;
              sub_70FDD0((__int64)v54, (__int64)a2, *a1, ((*((_BYTE *)a1 + 27) >> 1) ^ 1) & 1);
              *v8 = 1;
            }
            else
            {
              v11 = sub_8D5CE0(*v15, *a1);
              sub_710650(
                v54,
                v11,
                *a1,
                a2,
                0,
                0,
                (*((_BYTE *)a1 + 27) & 2) != 0,
                (v47 >> 1) & 1,
                0,
                &v53,
                dword_4F07508,
                v55);
              if ( LODWORD(v55[0]) | v53 )
              {
                sub_724E30(&v54);
                return v13;
              }
            }
          }
          else if ( v16 > 6u )
          {
            if ( word_4D04898 && (*((_BYTE *)a1 + 58) & 2) != 0 && (!dword_4F077BC || qword_4F077A8 <= 0x9E97u)
              || !(unsigned int)sub_716BB0(v15, v54, v50, v47, v8) )
            {
              goto LABEL_27;
            }
            v40 = sub_72D2E0(*a1, 0);
            if ( (unsigned int)sub_8DBE70(*a1) || *v8 )
            {
              v11 = (__int64)a2;
              sub_70FDD0((__int64)v54, (__int64)a2, v40, ((*((_BYTE *)a1 + 27) >> 1) ^ 1) & 1);
              *v8 = 1;
            }
            else
            {
              sub_72A510(v54, a2);
              v11 = v40;
              sub_70C9E0((__int64)a2, v40, (*((_BYTE *)a1 + 27) & 2) != 0, v41, v42);
            }
            if ( (*((_BYTE *)a1 + 58) & 2) != 0 )
              a2[10].m128i_i8[8] |= 0x40u;
          }
          else
          {
            if ( v16 != 3 )
            {
              if ( v16 == 4
                && (unsigned int)sub_8D2FB0(*v15)
                && (unsigned int)sub_7164A0(v15, (__int64)v54, v50, v47, (int *)v8) )
              {
                sub_72A510(v54, a2);
                v36 = sub_8D46C0(*v15);
                v11 = 0;
                v37 = sub_72D2E0(v36, 0);
                a2[9].m128i_i64[0] = 0;
                a2[8].m128i_i64[0] = v37;
                goto LABEL_71;
              }
LABEL_27:
              v13 = 0;
              sub_724E30(&v54);
              return v13;
            }
            if ( !(unsigned int)sub_8D2E30(*v15) )
              goto LABEL_27;
            v11 = (__int64)a2;
            if ( !(unsigned int)sub_7164A0(v15, (__int64)a2, v50, v47, (int *)v8) )
              goto LABEL_27;
          }
LABEL_71:
          v9 = (__int64)&v54;
          sub_724E30(&v54);
          if ( v8 == &v52 )
            goto LABEL_36;
          return 1;
        }
        switch ( v16 )
        {
          case '\\':
            v11 = (__int64)a2;
            if ( !(unsigned int)sub_716A40((__int64)a1, a2, v50, v47, (int *)v8, v17) )
              goto LABEL_27;
            goto LABEL_71;
          case '^':
          case '`':
            if ( (*((_BYTE *)v15 + 25) & 3) != 0 )
            {
              v24 = (__int64)v15;
              if ( (unsigned int)sub_716BB0(v15, v54, v50, v47, v8) )
                goto LABEL_67;
              if ( !word_4D04898 )
                goto LABEL_27;
              v18 = v54;
              if ( (*((_BYTE *)v15 + 25) & 3) != 0 )
              {
                if ( dword_4D03F94 )
                  goto LABEL_27;
                v24 = (__int64)v15;
                if ( !(unsigned int)sub_70D5B0((__int64)v15, (__int64)v54) )
                  goto LABEL_27;
                goto LABEL_67;
              }
            }
            else if ( !word_4D04898 )
            {
              goto LABEL_27;
            }
            if ( !(unsigned int)sub_716120((__int64)v15, (__int64)v18) )
              goto LABEL_27;
            v24 = sub_73A460(v18);
            sub_72D460(v24, v18);
LABEL_67:
            if ( v16 == 94 )
            {
LABEL_68:
              v30 = *(_QWORD *)(v46 + 56);
              if ( (*(_BYTE *)(v30 + 144) & 4) != 0 && !(unsigned int)sub_6ECD90(*(_QWORD *)(v46 + 56)) )
                goto LABEL_27;
              v31 = sub_72D2E0(*a1, 0);
              v11 = v30;
              if ( !(unsigned int)sub_7161E0(v54, v30, v31, a2, v32, v33) )
                goto LABEL_27;
              goto LABEL_71;
            }
LABEL_58:
            v55[0] = sub_724DC0(v24, v18, v25, v26, v27, v28);
            if ( (unsigned int)sub_716120(v46, v55[0]) )
            {
              v29 = v55[0];
            }
            else
            {
              if ( *(_BYTE *)(v46 + 24) != 2 )
                goto LABEL_62;
              v29 = *(_QWORD *)(v46 + 56);
            }
            if ( v29 )
            {
              if ( *(_BYTE *)(v29 + 173) == 7 && (*(_BYTE *)(v29 + 192) & 2) == 0 )
              {
                if ( *(_QWORD *)(v29 + 200) )
                {
                  v43 = sub_72D2E0(*a1, 0);
                  v11 = *(_QWORD *)(v29 + 200);
                  if ( (unsigned int)sub_7161E0(v54, v11, v43, a2, v44, v45) )
                  {
                    sub_724E30(v55);
                    goto LABEL_71;
                  }
                }
              }
            }
LABEL_62:
            v13 = 0;
            sub_724E30(v55);
            sub_724E30(&v54);
            return v13;
          case '_':
          case 'a':
            v23 = v47 | 2;
            if ( (*((_BYTE *)a1 + 25) & 1) != 0 )
              v23 = v47 | 6;
            v48 = v23;
            if ( !(unsigned int)sub_8D2E30(*v15) || (*((_BYTE *)a1 + 25) & 1) != 0 && *((_BYTE *)v15 + 24) == 24 )
              goto LABEL_27;
            v18 = v54;
            v24 = (__int64)v15;
            if ( !(unsigned int)sub_7164A0(v15, (__int64)v54, v50, v48, (int *)v8) )
              goto LABEL_27;
            if ( v16 != 95 )
              goto LABEL_58;
            goto LABEL_68;
          case 't':
            if ( *((_BYTE *)v15 + 24) != 2 )
              goto LABEL_27;
            v34 = v15[7];
            if ( *(_BYTE *)(v34 + 173) != 12 )
              goto LABEL_27;
            v35 = *(_BYTE *)(v34 + 176);
            if ( v35 == 11 )
              v35 = *(_BYTE *)(*(_QWORD *)(v34 + 184) + 176LL);
            if ( v35 == 2 )
            {
              sub_724C70(a2, 12);
              v11 = 4;
              sub_7249B0(a2, 4);
              a2[11].m128i_i64[1] = v34;
              a2[8].m128i_i64[0] = dword_4D03B80;
              *v8 = 1;
            }
            else
            {
              if ( v35 != 3 )
                goto LABEL_27;
              v11 = (__int64)a2;
              sub_72A510(v15[7], a2);
              *v8 = 1;
            }
            goto LABEL_71;
          default:
            goto LABEL_27;
        }
      case 2:
        v19 = a1[7];
        v20 = *(_BYTE *)(v19 + 173);
        if ( v20 == 2 )
        {
          v11 = (__int64)a2;
          v9 = a1[7];
          sub_72D410(v9, a2);
          goto LABEL_18;
        }
        if ( v20 != 6 )
        {
          if ( v20 != 12 )
            return 0;
          v38 = *a1;
          if ( (unsigned int)sub_8D3410(v38) )
            v38 = sub_8D4050(v38);
          v39 = sub_72D2E0(v38, 0);
          v11 = (__int64)a2;
          v9 = v19;
          sub_70FDD0(v19, (__int64)a2, v39, 0);
          *v8 = 1;
          goto LABEL_18;
        }
        v11 = (__int64)a2;
        v9 = a1[7];
        sub_72A510(v9, a2);
        if ( v8 != &v52 )
          return 1;
        goto LABEL_36;
      case 3:
        v21 = a1[7];
        if ( *(_BYTE *)(v21 + 177) == 5 )
        {
          a1 = *(_QWORD **)(v21 + 184);
          continue;
        }
        v51 = a3;
        v49 = a4;
        if ( sub_6EA100((_BYTE *)v21) )
        {
          if ( *(_BYTE *)(v21 + 136) == 3 )
            goto LABEL_112;
          v11 = (__int64)a2;
          v9 = v21;
          sub_72D510(v21, a2, v51);
LABEL_113:
          if ( (*(_BYTE *)(v21 + 89) & 4) != 0
            && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v21 + 40) + 32LL) + 177LL) & 0x20) != 0
            || (v9 = *(_QWORD *)(v21 + 120), (unsigned int)sub_8DBE70(v9)) )
          {
            *v8 = 1;
            if ( v8 == &v52 )
              goto LABEL_36;
          }
          else
          {
LABEL_18:
            if ( v8 == &v52 )
            {
LABEL_36:
              v13 = 1;
              goto LABEL_37;
            }
          }
          return 1;
        }
        if ( (v49 & 1) != 0 && *(_BYTE *)(v21 + 136) == 3 )
        {
LABEL_112:
          *(_BYTE *)(v21 + 136) = 2;
          v11 = (__int64)a2;
          v9 = v21;
          sub_72D510(v21, a2, v51);
          *(_BYTE *)(v21 + 136) = 3;
          goto LABEL_113;
        }
        return 0;
      case 11:
        v22 = a1[7];
        if ( *(_QWORD *)(v22 + 16) )
          return 0;
        v9 = *(_QWORD *)(v22 + 56);
        v11 = 0;
        sub_73C780(v9, 0, a2);
        if ( v8 == &v52 )
          goto LABEL_36;
        return 1;
      case 20:
        v9 = a1[7];
        if ( (*(_BYTE *)(v9 + 193) & 4) != 0 )
          return 0;
        v11 = (__int64)a2;
        sub_70D050(v9, (__int64)a2, a3, v8);
        if ( v8 == &v52 )
          goto LABEL_36;
        return 1;
      default:
        v13 = 0;
        goto LABEL_10;
    }
  }
}
