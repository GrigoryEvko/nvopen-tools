// Function: sub_7CBA40
// Address: 0x7cba40
//
__int64 __fastcall sub_7CBA40(int a1, _DWORD *a2, _QWORD *a3)
{
  unsigned __int64 v3; // r10
  unsigned __int64 v5; // rax
  char v6; // si
  int v7; // r9d
  int v8; // r8d
  int v9; // r14d
  int v10; // r13d
  char v12; // dl
  int v13; // ebx
  const char *v14; // r14
  unsigned __int8 v15; // di
  int v16; // eax
  unsigned __int64 v17; // r15
  unsigned __int8 v18; // r12
  int v19; // r14d
  const char *v20; // rbx
  unsigned __int8 v21; // al
  unsigned __int8 v22; // r14
  __int64 result; // rax
  int v24; // r15d
  _QWORD *v25; // rax
  const char *v26; // r14
  unsigned __int64 v27; // r15
  const char *v28; // rax
  unsigned __int64 v29; // rsi
  __int64 v30; // rdx
  unsigned __int8 v31; // dl
  unsigned __int8 v32; // bl
  bool v33; // al
  const char *v34; // r13
  int v35; // r8d
  unsigned __int64 v36; // r15
  unsigned __int8 v37; // di
  _BOOL4 v38; // eax
  char v39; // si
  int v40; // [rsp+Ch] [rbp-C4h]
  int v41; // [rsp+10h] [rbp-C0h]
  int v42; // [rsp+18h] [rbp-B8h]
  int v43; // [rsp+18h] [rbp-B8h]
  int v44; // [rsp+1Ch] [rbp-B4h]
  int v45; // [rsp+20h] [rbp-B0h]
  bool v46; // [rsp+27h] [rbp-A9h]
  unsigned __int64 v48; // [rsp+30h] [rbp-A0h]
  unsigned __int64 v49; // [rsp+30h] [rbp-A0h]
  unsigned __int64 v50; // [rsp+30h] [rbp-A0h]
  unsigned __int64 v51; // [rsp+30h] [rbp-A0h]
  const char *v53; // [rsp+40h] [rbp-90h]
  int v55; // [rsp+5Ch] [rbp-74h] BYREF
  __m128i v56; // [rsp+60h] [rbp-70h] BYREF
  __m128i v57; // [rsp+70h] [rbp-60h] BYREF
  __int16 v58[8]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v59[32]; // [rsp+90h] [rbp-40h] BYREF

  v3 = qword_4F06408;
  *a2 = 0;
  v46 = a1 != 10;
  if ( (unsigned __int64)qword_4F06410 > v3 )
  {
    v45 = 0;
    v9 = 0;
    v10 = 0;
    v44 = 0;
    if ( a1 != 10 )
      goto LABEL_17;
LABEL_51:
    v13 = unk_4F061E0;
    if ( !unk_4F061E0 && (__int64)(v3 - (_QWORD)qword_4F06410) <= 18 )
    {
      v28 = qword_4F06410 + 1;
      v29 = *qword_4F06410 - 48;
      if ( v3 >= (unsigned __int64)(qword_4F06410 + 1) )
      {
        do
        {
          v30 = *v28++;
          v29 = v30 + 10 * v29 - 48;
        }
        while ( v28 != (const char *)(v3 + 1) );
      }
      sub_620DE0(&v56, v29);
      goto LABEL_29;
    }
    v49 = v3;
    sub_620DE0(&v57, 0xAu);
    sub_620DE0(&v56, *qword_4F06410 - 48);
    v53 = (const char *)(v49 + 1);
    if ( v49 >= (unsigned __int64)(qword_4F06410 + 1) )
    {
      v40 = v9;
      v19 = 0;
      v20 = qword_4F06410 + 1;
      do
      {
        if ( *v20 != 39 )
        {
          v50 = *v20 - 48;
          sub_621F20(&v56, &v57, 0, (_BOOL4 *)&v55);
          if ( v55 )
            v19 = 1;
          sub_620DE0(v58, v50);
          sub_621270((unsigned __int16 *)&v56, v58, 0, (_BOOL4 *)&v55);
          if ( v55 )
            v19 = 1;
        }
        ++v20;
      }
      while ( v20 != v53 );
      v13 = v19;
      v9 = v40;
      goto LABEL_29;
    }
    goto LABEL_124;
  }
  v5 = v3 - 1;
  v6 = 0;
  v7 = 0;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  while ( 1 )
  {
    v3 = v5 + 1;
    v12 = *(_BYTE *)(v5 + 1) & 0xDF;
    if ( v12 == 85 )
    {
      v10 = 1;
      goto LABEL_4;
    }
    if ( v12 != 76 )
      break;
    if ( v9 )
    {
      if ( *(_BYTE *)(v5 + 1) == v6 || !dword_4D04964 )
      {
        v7 = v9;
        v9 = 0;
      }
      else
      {
        *a3 = v3;
        v7 = v9;
        v9 = 0;
        *a2 = 886;
      }
    }
    else
    {
      v6 = *(_BYTE *)(v5 + 1);
      v9 = 1;
    }
LABEL_4:
    --v5;
  }
  if ( unk_4D04184 && v12 == 90 )
  {
    v8 = 1;
    goto LABEL_4;
  }
  v45 = v7;
  v44 = v8;
  if ( a1 == 10 )
    goto LABEL_51;
LABEL_17:
  if ( a1 != 8 )
  {
    v48 = v3;
    if ( a1 == 2 )
    {
      sub_620DE0(&v56, 0);
      if ( v48 >= (unsigned __int64)(qword_4F06410 + 2) )
      {
        v41 = v10;
        v13 = 0;
        v42 = v9;
        v26 = qword_4F06410 + 2;
        while ( 1 )
        {
          if ( *v26 != 39 )
          {
            v27 = *v26 - 48;
            if ( v27 > 1 )
            {
              *a3 = v26;
              *a2 = 2424;
              return sub_72C970((__int64)xmmword_4F06300);
            }
            sub_621410((__int64)&v56, 1, &v55);
            if ( v55 )
              v13 = 1;
            sub_620DE0(v58, v27);
            sub_6213B0((__int64)&v56, (__int64)v58);
          }
          if ( (const char *)(v48 + 1) == ++v26 )
            goto LABEL_28;
        }
      }
    }
    else
    {
      sub_620DE0(&v56, 0);
      if ( v48 >= (unsigned __int64)(qword_4F06410 + 2) )
      {
        v41 = v10;
        v13 = 0;
        v42 = v9;
        v14 = qword_4F06410 + 2;
        do
        {
          v15 = *v14;
          if ( *v14 != 39 )
          {
            v16 = 48;
            if ( (unsigned int)v15 - 48 > 9 )
              v16 = islower(v15) == 0 ? 55 : 87;
            v17 = (char)v15 - v16;
            sub_621410((__int64)&v56, 4, &v55);
            if ( v55 )
              v13 = 1;
            sub_620DE0(v58, v17);
            sub_6213B0((__int64)&v56, (__int64)v58);
          }
          ++v14;
        }
        while ( (const char *)(v48 + 1) != v14 );
LABEL_28:
        v10 = v41;
        v9 = v42;
        goto LABEL_29;
      }
    }
LABEL_124:
    v13 = 0;
    goto LABEL_29;
  }
  v51 = v3;
  sub_620DE0(&v56, 0);
  if ( v51 < (unsigned __int64)(qword_4F06410 + 1) )
    goto LABEL_124;
  v13 = 0;
  v43 = v10;
  v34 = qword_4F06410 + 1;
  do
  {
    if ( *v34 != 39 )
    {
      v35 = *v34 - 48;
      v36 = v35;
      if ( dword_4F077C4 != 1 && (unsigned __int64)v35 > 7 )
      {
        *a3 = v34;
        *a2 = 24;
        return sub_72C970((__int64)xmmword_4F06300);
      }
      sub_621410((__int64)&v56, 3, &v55);
      if ( v55 )
        v13 = 1;
      sub_620DE0(v58, v36);
      sub_6213B0((__int64)&v56, (__int64)v58);
    }
    ++v34;
  }
  while ( v34 != (const char *)(v51 + 1) );
  v10 = v43;
LABEL_29:
  if ( !unk_4D03D04 )
  {
LABEL_33:
    if ( dword_4F077C4 != 1 || v10 | v45 )
    {
LABEL_34:
      if ( v13 )
        goto LABEL_66;
      if ( !v44 )
      {
        if ( v45 )
          goto LABEL_42;
        if ( v9 )
        {
          if ( v10 )
            goto LABEL_40;
          if ( !sub_6211E0(v56.m128i_i16, 0, 7u) )
          {
            if ( a1 != 10 )
              goto LABEL_40;
            goto LABEL_133;
          }
        }
        else
        {
          if ( v10 )
          {
            if ( !sub_6211E0(v56.m128i_i16, 0, 6u) )
              goto LABEL_40;
LABEL_137:
            v18 = 6;
            goto LABEL_109;
          }
          if ( sub_6211E0(v56.m128i_i16, 0, 5u) )
            goto LABEL_158;
          if ( a1 == 10 )
          {
            if ( !sub_6211E0(v56.m128i_i16, 0, 7u) )
            {
LABEL_133:
              if ( unk_4D04294 )
              {
                if ( !dword_4D04964 )
                  goto LABEL_156;
LABEL_135:
                if ( !unk_4D04298 )
                  goto LABEL_102;
LABEL_42:
                if ( v10 )
                {
LABEL_43:
                  if ( dword_4F077C4 != 2 && unk_4F07778 > 199900 && !v10 && !v46 )
                    goto LABEL_47;
                  if ( sub_6211E0(v56.m128i_i16, 0, 0xAu) )
                  {
LABEL_129:
                    v18 = 10;
                    goto LABEL_109;
                  }
                  if ( dword_4F077C4 != 2 && unk_4F07778 > 199900 )
                  {
LABEL_47:
                    if ( unk_4D04290 )
                      goto LABEL_103;
                    if ( sub_6211E0(v56.m128i_i16, 0, 0xAu) )
                    {
                      sub_7B0EB0((unsigned __int64)qword_4F06410, (__int64)dword_4F07508);
                      v18 = 10;
                      sub_684B30(0x511u, dword_4F07508);
LABEL_109:
                      sub_724C70((__int64)xmmword_4F06300, 1);
                      xmmword_4F06380[0].m128i_i64[0] = (__int64)sub_72BA30(v18);
                      goto LABEL_110;
                    }
                  }
LABEL_102:
                  if ( !unk_4D04290 )
                    goto LABEL_66;
LABEL_103:
                  if ( (_DWORD)qword_4F077B4 )
                  {
                    v37 = 7;
                  }
                  else
                  {
                    if ( !HIDWORD(qword_4F077B4) )
                    {
                      if ( !v10 && sub_6211E0(v56.m128i_i16, 0, 0xBu) )
                      {
                        v18 = 11;
                      }
                      else
                      {
                        if ( !sub_6211E0(v56.m128i_i16, 0, 0xCu) )
                          goto LABEL_66;
                        v18 = 12;
                      }
                      goto LABEL_109;
                    }
                    v37 = 5;
                  }
                  sub_684AA0(v37, 0x17u, dword_4F07508);
                  if ( (_DWORD)qword_4F077B4 )
                    goto LABEL_129;
LABEL_158:
                  v18 = 5;
                  goto LABEL_109;
                }
LABEL_156:
                if ( sub_6211E0(v56.m128i_i16, 0, 9u) )
                {
                  v18 = 9;
                  goto LABEL_109;
                }
                goto LABEL_43;
              }
              goto LABEL_40;
            }
          }
          else
          {
            if ( sub_6211E0(v56.m128i_i16, 0, 6u) )
              goto LABEL_137;
            if ( !sub_6211E0(v56.m128i_i16, 0, 7u) )
            {
LABEL_40:
              if ( sub_6211E0(v56.m128i_i16, 0, 8u) )
              {
                v18 = 8;
                goto LABEL_109;
              }
              if ( !dword_4D04964 )
                goto LABEL_42;
              goto LABEL_135;
            }
          }
        }
        v18 = 7;
        goto LABEL_109;
      }
      v31 = byte_4F06A51[0];
      if ( byte_4B6DF90[byte_4F06A51[0]] )
      {
        v31 = byte_4F06A51[0] + 1;
        v32 = byte_4F06A51[0];
      }
      else
      {
        v32 = byte_4F06A51[0] - 1;
      }
      v9 = v10 ^ 1;
      if ( v10 )
      {
        v32 = v31;
      }
      else if ( a1 == 10 )
      {
        v22 = v32;
        if ( !sub_6211E0(v56.m128i_i16, 1, v32) )
          goto LABEL_66;
        goto LABEL_98;
      }
      LOBYTE(v24) = v32;
      if ( !sub_6211E0(v56.m128i_i16, 0, v31) )
        goto LABEL_66;
LABEL_76:
      sub_724C70((__int64)xmmword_4F06300, 1);
      v25 = sub_72BA30(v24);
      xmmword_4F06380[0].m128i_i64[0] = (__int64)v25;
      if ( v9 )
        sub_6215A0(v56.m128i_i16, *((_DWORD *)v25 + 32) * dword_4F06BA0);
      goto LABEL_110;
    }
    if ( v9 )
    {
      if ( !sub_6211E0(v56.m128i_i16, 0, 8u) )
      {
        if ( !sub_6211E0(v56.m128i_i16, 0, 0xAu) )
        {
          v24 = 9;
          goto LABEL_147;
        }
        if ( a1 != 10 )
        {
          v24 = 9;
          goto LABEL_75;
        }
        goto LABEL_173;
      }
      if ( a1 != 10 )
      {
        v24 = 7;
        goto LABEL_75;
      }
    }
    else
    {
      if ( a1 != 10 )
      {
        if ( sub_6211E0(v56.m128i_i16, 0, 6u) )
        {
          v24 = 5;
          v9 = 1;
        }
        else
        {
          if ( sub_6211E0(v56.m128i_i16, 0, 8u) )
          {
            v24 = 7;
            v9 = 1;
            goto LABEL_75;
          }
          v24 = 9;
          v9 = 1;
          if ( !sub_6211E0(v56.m128i_i16, 0, 0xAu) )
          {
LABEL_147:
            sub_7B0EB0((unsigned __int64)qword_4F06410, (__int64)dword_4F07508);
            sub_684B30(0x17u, dword_4F07508);
            v9 = byte_4B6DF90[v24];
            sub_621EE0(v59, unk_4F06AE0 * dword_4F06BA0);
            sub_6213D0((__int64)&v56, (__int64)v59);
            goto LABEL_76;
          }
        }
LABEL_75:
        if ( !v13 )
          goto LABEL_76;
        goto LABEL_147;
      }
      if ( sub_6211E0(v56.m128i_i16, 0, 5u) )
      {
        v46 = 0;
        v24 = 5;
        goto LABEL_75;
      }
      if ( !sub_6211E0(v56.m128i_i16, 0, 8u) )
      {
        if ( !sub_6211E0(v56.m128i_i16, 0, 0xAu) )
        {
          v46 = 0;
          v24 = 9;
          goto LABEL_147;
        }
LABEL_173:
        v24 = 9;
        v38 = sub_6211E0(v56.m128i_i16, 0, 9u);
        goto LABEL_169;
      }
    }
    v24 = 7;
    v38 = sub_6211E0(v56.m128i_i16, 0, 7u);
LABEL_169:
    v39 = v46;
    v9 = 1;
    if ( !v38 )
      v39 = 1;
    v46 = v39;
    goto LABEL_75;
  }
  if ( dword_4F077C4 == 2 )
  {
    if ( HIDWORD(qword_4F077B4) )
      goto LABEL_63;
    goto LABEL_34;
  }
  if ( unk_4F07778 <= 199900 && !HIDWORD(qword_4F077B4) )
    goto LABEL_33;
LABEL_63:
  if ( !v10 && (a1 == 10 || sub_6211E0(v56.m128i_i16, 0, unk_4F06AC9)) )
    v21 = unk_4F06AC9;
  else
    v21 = unk_4F06AC8;
  v22 = v21;
  if ( v13 )
  {
LABEL_66:
    *a3 = qword_4F06410;
    *a2 = 23;
    return sub_72C970((__int64)xmmword_4F06300);
  }
LABEL_98:
  sub_724C70((__int64)xmmword_4F06300, 1);
  xmmword_4F06380[0].m128i_i64[0] = (__int64)sub_72BA30(v22);
LABEL_110:
  *(__m128i *)word_4F063B0 = _mm_loadu_si128(&v56);
  byte_4F063A9[0] = v46 | byte_4F063A9[0] & 0xFE;
  v33 = 0;
  if ( qword_4F06410 == (const char *)qword_4F06408 )
    v33 = *qword_4F06410 == 48;
  byte_4F063A9[0] = (2 * v33) | byte_4F063A9[0] & 0xFD;
  result = (__int64)a2;
  if ( *a2 )
    return sub_72C970((__int64)xmmword_4F06300);
  return result;
}
