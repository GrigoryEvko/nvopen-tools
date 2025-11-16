// Function: sub_7D8470
// Address: 0x7d8470
//
_QWORD *__fastcall sub_7D8470(__int64 *a1)
{
  __m128i *v2; // r15
  __int64 i; // r13
  __int64 j; // r12
  _QWORD *result; // rax
  __int64 v6; // rcx
  __int64 v7; // r8
  const __m128i *v8; // rsi
  char v9; // dl
  char v10; // al
  __int64 v11; // r8
  char **v12; // rax
  __int64 v13; // r13
  __int128 *v14; // rbx
  __int64 v15; // rdi
  _QWORD *v16; // rax
  _QWORD *v17; // rdi
  __int64 v18; // r8
  char **v19; // rax
  int v20; // eax
  int v21; // ebx
  __int64 v22; // r13
  char **v23; // rax
  _QWORD *v24; // rax
  _BYTE *v25; // rsi
  __int128 *v26; // rax
  __int64 v27; // rdi
  __int128 *v28; // rax
  char **v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // rsi
  char **v32; // rcx
  _QWORD *v33; // rsi
  _QWORD *v34; // rdi
  const __m128i *v35; // rsi
  __int64 *v36; // rax
  const __m128i *v37; // rax
  char **v38; // rax
  _QWORD *v39; // rax
  char *v40; // [rsp+0h] [rbp-50h]
  __int64 v41; // [rsp+8h] [rbp-48h]
  char *v42; // [rsp+8h] [rbp-48h]
  const __m128i *v43; // [rsp+18h] [rbp-38h] BYREF

  v2 = (__m128i *)a1[9];
  for ( i = v2->m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  for ( j = *a1; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  result = (_QWORD *)sub_8D2600(j);
  if ( (_DWORD)result )
    return result;
  if ( i == j || (unsigned int)sub_8D97D0(i, j, 1, v6, v7) )
  {
    result = (_QWORD *)sub_8D2B20(i);
    if ( (_DWORD)result )
      return result;
    v8 = v2;
    return (_QWORD *)sub_730620((__int64)a1, v8);
  }
  if ( (unsigned int)sub_8D2B50(i) && (unsigned int)sub_8D2B50(j) )
  {
    v9 = *(_BYTE *)(i + 160);
    v10 = *(_BYTE *)(j + 160);
    switch ( v9 )
    {
      case 10:
        if ( v10 == 10 )
          goto LABEL_24;
        v9 = 0;
        if ( v10 == 11 )
          goto LABEL_27;
LABEL_20:
        if ( v10 == 12 )
        {
          v10 = 4;
        }
        else if ( v10 == 13 )
        {
          v10 = 8;
        }
LABEL_23:
        if ( v10 == v9 )
        {
LABEL_24:
          result = (_QWORD *)sub_730620((__int64)a1, v2);
          *a1 = j;
          return result;
        }
        goto LABEL_27;
      case 11:
        v9 = 2;
        break;
      case 12:
        v9 = 4;
        break;
      case 13:
        v9 = 8;
        break;
      default:
        if ( v10 == 10 )
        {
          v10 = 0;
          goto LABEL_23;
        }
        goto LABEL_19;
    }
    if ( v10 != 10 )
    {
LABEL_19:
      if ( v10 == 11 )
      {
        v10 = 2;
        goto LABEL_23;
      }
      goto LABEL_20;
    }
  }
LABEL_27:
  if ( (unsigned int)sub_8D2B50(j) )
  {
    if ( (unsigned int)sub_8D2B50(i) )
    {
      switch ( *(_BYTE *)(i + 160) )
      {
        case 0:
          v28 = &xmmword_4F18340;
          v29 = (char **)&unk_4B7B5A0;
          goto LABEL_59;
        case 2:
          v28 = &xmmword_4F18280;
          v29 = &off_4B7B4E0;
          goto LABEL_59;
        case 3:
        case 4:
          v28 = &xmmword_4F18220;
          v29 = &off_4B7B480;
          goto LABEL_59;
        case 5:
        case 6:
          v28 = &xmmword_4F181C0;
          v29 = &off_4B7B420;
          goto LABEL_59;
        case 7:
          v28 = &xmmword_4F18160;
          v29 = &off_4B7B3C0;
          goto LABEL_59;
        case 8:
          v28 = &xmmword_4F18100;
          v29 = &off_4B7B360;
          goto LABEL_59;
        case 9:
          v28 = &xmmword_4F182E0;
          v29 = off_4B7B540;
LABEL_59:
          v30 = *(unsigned __int8 *)(j + 160);
          v31 = 8 * v30;
          if ( (_BYTE)v30 != 10 )
          {
            if ( (_BYTE)v30 == 11 )
            {
              v29 += 2;
            }
            else if ( (_BYTE)v30 == 12 )
            {
              v29 += 4;
            }
            else
            {
              v32 = v29 + 8;
              v29 = (char **)((char *)v29 + v31);
              if ( (_BYTE)v30 == 13 )
                v29 = v32;
            }
          }
          v33 = (_QWORD *)((char *)v28 + v31);
          if ( *v33 )
            v8 = (const __m128i *)sub_7F88E0(*v33, v2);
          else
            v8 = (const __m128i *)sub_7F8B20(*v29, v33, j, v2->m128i_i64[0], 0, v2);
          return (_QWORD *)sub_730620((__int64)a1, v8);
        default:
          goto LABEL_72;
      }
    }
    v20 = sub_8D2B20(i);
    v21 = *(unsigned __int8 *)(j + 160);
    v22 = *(unsigned __int8 *)(j + 160);
    if ( v20 )
    {
      v23 = &off_4B7B720;
      if ( (_BYTE)v21 != 10 )
      {
        v23 = off_4B7B730;
        if ( (_BYTE)v21 != 11 )
        {
          v23 = off_4B7B740;
          if ( (_BYTE)v21 != 12 )
          {
            v23 = off_4B7B760;
            if ( (_BYTE)v21 != 13 )
              v23 = &(&off_4B7B720)[(unsigned __int8)v21];
          }
        }
      }
      v42 = *v23;
      v24 = sub_72C7D0(*(_BYTE *)(j + 160));
      v25 = sub_73E130(v2, (__int64)v24);
      v26 = &xmmword_4F18400;
    }
    else
    {
      v38 = &off_4B7B660;
      if ( (_BYTE)v21 != 10 )
      {
        v38 = off_4B7B670;
        if ( (_BYTE)v21 != 11 )
        {
          v38 = off_4B7B680;
          if ( (_BYTE)v21 != 12 )
          {
            v38 = off_4B7B6A0;
            if ( (_BYTE)v21 != 13 )
              v38 = &(&off_4B7B660)[(unsigned __int8)v21];
          }
        }
      }
      v42 = *v38;
      v39 = sub_72C610(*(_BYTE *)(j + 160));
      v25 = sub_73E130(v2, (__int64)v39);
      v26 = &xmmword_4F184C0;
    }
    v27 = *((_QWORD *)v26 + v21);
    if ( v27 )
      v8 = (const __m128i *)sub_7F88E0(v27, v25);
    else
      v8 = (const __m128i *)sub_7F8B20(v42, (char *)v26 + 8 * v22, j, *(_QWORD *)v25, 0, v25);
    return (_QWORD *)sub_730620((__int64)a1, v8);
  }
  if ( !(unsigned int)sub_8D2B20(j) )
  {
    if ( (unsigned int)sub_8D2B50(i) )
    {
      v18 = *(unsigned __int8 *)(i + 160);
      v19 = &off_4B7B600;
      v13 = v18;
      if ( (_BYTE)v18 != 10 )
      {
        v19 = off_4B7B610;
        if ( (_BYTE)v18 != 11 )
        {
          v19 = off_4B7B620;
          if ( (_BYTE)v18 != 12 )
          {
            v19 = off_4B7B640;
            if ( (_BYTE)v18 != 13 )
              v19 = &(&off_4B7B600)[(unsigned __int8)v18];
          }
        }
      }
      v14 = &xmmword_4F18460;
      v15 = *((_QWORD *)&xmmword_4F18460 + (int)v18);
      if ( !v15 )
      {
        v40 = *v19;
        v41 = v2->m128i_i64[0];
        v16 = sub_72C610(v18);
        goto LABEL_39;
      }
LABEL_47:
      v17 = (_QWORD *)sub_7F88E0(v15, v2);
      goto LABEL_48;
    }
    if ( !(unsigned int)sub_8D2B20(i) )
LABEL_72:
      sub_721090();
    v34 = (_QWORD *)j;
    v43 = (const __m128i *)sub_724DC0();
    v35 = v43;
LABEL_79:
    sub_72BB40((__int64)v34, v35);
    v36 = sub_73A720(v43, (__int64)v35);
    v37 = (const __m128i *)sub_73DF90((__int64)v2, v36);
    sub_730620((__int64)a1, v37);
    return sub_724E30((__int64)&v43);
  }
  if ( (unsigned int)sub_8D2B50(i) )
  {
    v11 = *(unsigned __int8 *)(i + 160);
    v12 = &off_4B7B6C0;
    v13 = v11;
    if ( (_BYTE)v11 != 10 )
    {
      v12 = off_4B7B6D0;
      if ( (_BYTE)v11 != 11 )
      {
        v12 = off_4B7B6E0;
        if ( (_BYTE)v11 != 12 )
        {
          v12 = off_4B7B700;
          if ( (_BYTE)v11 != 13 )
            v12 = &(&off_4B7B6C0)[(unsigned __int8)v11];
        }
      }
    }
    v14 = &xmmword_4F183A0;
    v15 = *((_QWORD *)&xmmword_4F183A0 + (int)v11);
    if ( !v15 )
    {
      v40 = *v12;
      v41 = v2->m128i_i64[0];
      v16 = sub_72C7D0(v11);
LABEL_39:
      v17 = (_QWORD *)sub_7F8B20(v40, (char *)v14 + 8 * v13, v16, v41, 0, v2);
LABEL_48:
      v8 = (const __m128i *)sub_73E130(v17, j);
      return (_QWORD *)sub_730620((__int64)a1, v8);
    }
    goto LABEL_47;
  }
  if ( (unsigned int)sub_8D2AC0(i) || (result = (_QWORD *)sub_8D2930(i), (_DWORD)result) )
  {
    v43 = (const __m128i *)sub_724DC0();
    v35 = v43;
    v34 = sub_72C610(*(_BYTE *)(j + 160));
    goto LABEL_79;
  }
  return result;
}
