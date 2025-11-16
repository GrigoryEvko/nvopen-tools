// Function: sub_25DD5F0
// Address: 0x25dd5f0
//
__int64 __fastcall sub_25DD5F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  signed __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rax
  _BYTE *v9; // rax
  int v10; // edx
  unsigned __int8 *v11; // rax
  __int64 v12; // rdx
  char *v13; // r13
  unsigned __int8 v14; // dl
  __int64 v15; // r14
  __int64 v16; // rbx
  _BYTE *v17; // rax
  unsigned __int8 *v18; // r15
  unsigned __int64 v19; // rdx
  __int64 v20; // rax
  unsigned __int8 *v21; // rdi
  __int64 v22; // rcx
  unsigned __int8 *v23; // r14
  unsigned __int8 v24; // dl
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v29; // rax
  __int64 v30; // rsi
  int v31; // edx
  unsigned __int8 *v32; // rax
  int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // rdx
  _QWORD *v36; // rdx
  _BYTE *i; // r15
  _BYTE **v38; // rdx
  _BYTE *v40; // [rsp+18h] [rbp-2A8h]
  unsigned __int8 *v41; // [rsp+18h] [rbp-2A8h]
  __m128i v42; // [rsp+20h] [rbp-2A0h] BYREF
  __m128i v43; // [rsp+30h] [rbp-290h] BYREF
  _BYTE *v44; // [rsp+40h] [rbp-280h] BYREF
  __int64 v45; // [rsp+48h] [rbp-278h]
  _BYTE v46[48]; // [rsp+50h] [rbp-270h] BYREF
  _BYTE *v47; // [rsp+80h] [rbp-240h] BYREF
  __int64 v48; // [rsp+88h] [rbp-238h]
  _BYTE v49[560]; // [rsp+90h] [rbp-230h] BYREF

  v6 = 0;
  v7 = *(_QWORD *)(a1 + 16);
  v47 = v49;
  v48 = 0x2000000000LL;
  v44 = v46;
  v45 = 0x600000000LL;
  v8 = v7;
  v42.m128i_i64[0] = a2;
  v42.m128i_i64[1] = a3;
  if ( v7 )
  {
    do
    {
      v8 = *(_QWORD *)(v8 + 8);
      ++v6;
    }
    while ( v8 );
    v9 = v46;
    if ( v6 > 6 )
    {
      sub_C8D5F0((__int64)&v44, v46, v6, 8u, a5, a6);
      v9 = &v44[8 * (unsigned int)v45];
    }
    do
    {
      v9 += 8;
      *((_QWORD *)v9 - 1) = *(_QWORD *)(v7 + 24);
      v7 = *(_QWORD *)(v7 + 8);
    }
    while ( v7 );
    v10 = v6 + v45;
    LODWORD(v6) = 0;
    LODWORD(v45) = v10;
    LODWORD(v11) = v10;
    if ( v10 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v12 = (unsigned int)v11;
          v11 = (unsigned __int8 *)(unsigned int)((_DWORD)v11 - 1);
          v13 = *(char **)&v44[8 * v12 - 8];
          LODWORD(v45) = (_DWORD)v11;
          v14 = *v13;
          if ( (unsigned __int8)*v13 > 0x1Cu )
            break;
          if ( v14 == 5 && *((_WORD *)v13 + 1) == 34 )
          {
            sub_25DC650((__int64)&v44, &v44[8 * (_QWORD)v11], *((_QWORD *)v13 + 2), 0);
            goto LABEL_62;
          }
LABEL_10:
          if ( !(_DWORD)v11 )
            goto LABEL_11;
        }
        if ( v14 == 62 )
        {
          v23 = (unsigned __int8 *)*((_QWORD *)v13 - 8);
          v24 = *v23;
          if ( *v23 > 0x15u )
            goto LABEL_33;
LABEL_41:
          LODWORD(v6) = 1;
          sub_B43D60(v13);
          LODWORD(v11) = v45;
          if ( !(_DWORD)v45 )
            break;
        }
        else
        {
          if ( v14 != 85 )
            goto LABEL_10;
          v22 = *((_QWORD *)v13 - 4);
          if ( !v22 )
            goto LABEL_10;
          if ( !*(_BYTE *)v22
            && *(_QWORD *)(v22 + 24) == *((_QWORD *)v13 + 10)
            && (*(_BYTE *)(v22 + 33) & 0x20) != 0
            && ((*(_DWORD *)(v22 + 36) - 243) & 0xFFFFFFFD) == 0 )
          {
            v23 = *(unsigned __int8 **)&v13[32 * (1LL - (*((_DWORD *)v13 + 1) & 0x7FFFFFF))];
            v24 = *v23;
            if ( *v23 <= 0x15u )
              goto LABEL_41;
LABEL_33:
            if ( v24 <= 0x1Cu )
              goto LABEL_10;
            v25 = *((_QWORD *)v23 + 2);
            if ( !v25 || *(_QWORD *)(v25 + 8) )
              goto LABEL_10;
            v26 = (unsigned int)v48;
            v27 = (unsigned int)v48 + 1LL;
            if ( v27 > HIDWORD(v48) )
            {
              sub_C8D5F0((__int64)&v47, v49, v27, 0x10u, a5, a6);
              v26 = (unsigned int)v48;
            }
            v11 = &v47[16 * v26];
            *(_QWORD *)v11 = v23;
            *((_QWORD *)v11 + 1) = v13;
            LODWORD(v11) = v45;
            LODWORD(v48) = v48 + 1;
            if ( !(_DWORD)v45 )
              break;
          }
          else
          {
            if ( *(_BYTE *)v22 )
              goto LABEL_10;
            v30 = *((_QWORD *)v13 + 10);
            if ( *(_QWORD *)(v22 + 24) != v30 )
              goto LABEL_10;
            if ( (*(_BYTE *)(v22 + 33) & 0x20) == 0 )
              goto LABEL_10;
            v31 = *(_DWORD *)(v22 + 36);
            if ( v31 != 238 && (unsigned int)(v31 - 240) > 1 )
              goto LABEL_10;
            v32 = sub_BD3990(*(unsigned __int8 **)&v13[32 * (1LL - (*((_DWORD *)v13 + 1) & 0x7FFFFFF))], v30);
            if ( *v32 == 3 && (v33 = v32[80] & 1) != 0 )
            {
              LODWORD(v6) = v33;
              sub_B43D60(v13);
              LODWORD(v11) = v45;
              if ( !(_DWORD)v45 )
                break;
            }
            else
            {
              v11 = sub_BD3990(*(unsigned __int8 **)&v13[32 * (1LL - (*((_DWORD *)v13 + 1) & 0x7FFFFFF))], v30);
              if ( *v11 > 0x1Cu )
              {
                v34 = *((_QWORD *)v11 + 2);
                if ( v34 )
                {
                  if ( !*(_QWORD *)(v34 + 8) )
                  {
                    v35 = (unsigned int)v48;
                    a5 = (unsigned int)v48 + 1LL;
                    if ( a5 > HIDWORD(v48) )
                    {
                      v41 = v11;
                      sub_C8D5F0((__int64)&v47, v49, (unsigned int)v48 + 1LL, 0x10u, a5, a6);
                      v35 = (unsigned int)v48;
                      v11 = v41;
                    }
                    v36 = &v47[16 * v35];
                    *v36 = v11;
                    v36[1] = v13;
                    LODWORD(v11) = v45;
                    LODWORD(v48) = v48 + 1;
                    goto LABEL_10;
                  }
                }
              }
LABEL_62:
              LODWORD(v11) = v45;
              if ( !(_DWORD)v45 )
                break;
            }
          }
        }
      }
    }
LABEL_11:
    if ( (_DWORD)v48 )
    {
      v15 = 0;
      v16 = 16 * ((unsigned int)(v48 - 1) + 1LL);
      while ( 1 )
      {
        v17 = &v47[v15];
        v18 = *(unsigned __int8 **)&v47[v15];
        v43 = _mm_load_si128(&v42);
        v19 = *v18;
        if ( (unsigned __int8)v19 > 0x15u )
          break;
LABEL_74:
        sub_B43D60(*((_QWORD **)v17 + 1));
        for ( i = *(_BYTE **)&v47[v15];
              !sub_D5CB60(i, (__int64 (__fastcall *)(__int64, __int64))sub_25DC1F0, (__int64)&v42);
              i = v40 )
        {
          if ( (i[7] & 0x40) != 0 )
          {
            v38 = (_BYTE **)*((_QWORD *)i - 1);
            v40 = *v38;
            if ( **v38 <= 0x1Cu )
              break;
          }
          else
          {
            v40 = *(_BYTE **)&i[-32 * (*((_DWORD *)i + 1) & 0x7FFFFFF)];
            if ( *v40 <= 0x1Cu )
              break;
          }
          sub_B43D60(i);
        }
        LODWORD(v6) = 1;
        sub_B43D60(i);
LABEL_13:
        v15 += 16;
        if ( v16 == v15 )
          goto LABEL_44;
      }
      while ( 1 )
      {
        v20 = *((_QWORD *)v18 + 2);
        if ( !v20 )
          goto LABEL_13;
        if ( *(_QWORD *)(v20 + 8) )
          goto LABEL_13;
        if ( (unsigned __int8)v19 <= 0x3Du )
        {
          v29 = 0x2000000400400000LL;
          if ( _bittest64(&v29, v19) )
            goto LABEL_13;
        }
        if ( sub_D5CB60(v18, (__int64 (__fastcall *)(__int64, __int64))sub_25DC1F0, (__int64)&v43) )
          goto LABEL_25;
        if ( (unsigned __int8)sub_B46970(v18) )
          goto LABEL_13;
        if ( *v18 == 63 )
        {
          if ( !(unsigned __int8)sub_B4DD90((__int64)v18) )
            goto LABEL_13;
          if ( (v18[7] & 0x40) == 0 )
            goto LABEL_66;
        }
        else
        {
          if ( (*((_DWORD *)v18 + 1) & 0x7FFFFFF) != 1 )
            goto LABEL_13;
          if ( (v18[7] & 0x40) == 0 )
          {
LABEL_66:
            v21 = &v18[-32 * (*((_DWORD *)v18 + 1) & 0x7FFFFFF)];
            goto LABEL_24;
          }
        }
        v21 = (unsigned __int8 *)*((_QWORD *)v18 - 1);
LABEL_24:
        v18 = *(unsigned __int8 **)v21;
        v19 = **(unsigned __int8 **)v21;
        if ( (unsigned __int8)v19 <= 0x15u )
        {
LABEL_25:
          v17 = &v47[v15];
          goto LABEL_74;
        }
      }
    }
  }
  else
  {
    LODWORD(v45) = 0;
  }
LABEL_44:
  sub_AD0030(a1);
  if ( v44 != v46 )
    _libc_free((unsigned __int64)v44);
  if ( v47 != v49 )
    _libc_free((unsigned __int64)v47);
  return (unsigned int)v6;
}
