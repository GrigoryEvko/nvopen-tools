// Function: sub_7CD8F0
// Address: 0x7cd8f0
//
_DWORD *__fastcall sub_7CD8F0(unsigned __int64 a1, _DWORD *a2, _QWORD *a3)
{
  char v3; // al
  char v4; // r12
  int v5; // r14d
  unsigned __int64 v6; // rbx
  int v7; // eax
  _DWORD *result; // rax
  unsigned int v9; // r13d
  __m128i v10; // xmm0
  _QWORD *v11; // [rsp+0h] [rbp-110h]
  bool v13; // [rsp+1Fh] [rbp-F1h]
  int v14; // [rsp+30h] [rbp-E0h]
  int v15; // [rsp+34h] [rbp-DCh]
  int v16; // [rsp+38h] [rbp-D8h]
  int v17; // [rsp+3Ch] [rbp-D4h]
  __int64 v18; // [rsp+40h] [rbp-D0h]
  int v19; // [rsp+48h] [rbp-C8h]
  int v20; // [rsp+5Ch] [rbp-B4h] BYREF
  unsigned __int64 v21; // [rsp+60h] [rbp-B0h] BYREF
  const char *v22; // [rsp+68h] [rbp-A8h] BYREF
  __m128i v23; // [rsp+70h] [rbp-A0h] BYREF
  __int16 v24[8]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v25[8]; // [rsp+90h] [rbp-80h] BYREF
  _QWORD v26[2]; // [rsp+A0h] [rbp-70h] BYREF
  int v27; // [rsp+B0h] [rbp-60h]
  __int64 v28; // [rsp+B8h] [rbp-58h]
  int v29; // [rsp+C8h] [rbp-48h]
  char v30; // [rsp+CCh] [rbp-44h]

  v3 = *qword_4F06410;
  if ( *qword_4F06410 == 85 )
  {
    v4 = 4;
    v5 = unk_4F06B68 * dword_4F06BA0;
    v13 = 0;
    v11 = sub_72C300();
    v16 = 0;
    v17 = 0;
    v22 = qword_4F06410 + 2;
    v18 = (1LL << ((unsigned __int8)v5 - 1)) | ((1LL << ((unsigned __int8)v5 - 1)) - 1);
    goto LABEL_6;
  }
  if ( v3 > 85 )
  {
    if ( v3 == 117 )
    {
      v13 = a1 > 1;
      if ( qword_4F06410[1] == 56 )
      {
        v22 = qword_4F06410 + 3;
        v5 = dword_4F06BA0;
        v18 = (1LL << ((unsigned __int8)dword_4F06BA0 - 1)) | ((1LL << ((unsigned __int8)dword_4F06BA0 - 1)) - 1);
        if ( dword_4F077C4 == 2 || unk_4F07778 <= 202310 )
        {
          if ( unk_4D041B4 )
          {
            v4 = 2;
            v16 = 1;
            v11 = sub_72C2A0();
            v17 = 0;
          }
          else
          {
            v4 = 0;
            v17 = unk_4F06B98;
            v16 = 1;
            v11 = sub_72BA30(0);
          }
        }
        else
        {
          v4 = 0;
          v16 = 1;
          v11 = sub_72BA30(2u);
          v17 = 0;
        }
      }
      else
      {
        v4 = 3;
        v18 = -1;
        v5 = 64;
        v11 = sub_72C2D0();
        v16 = 0;
        v17 = 0;
        v22 = qword_4F06410 + 2;
      }
      goto LABEL_6;
    }
    goto LABEL_89;
  }
  if ( v3 == 39 )
  {
    v22 = qword_4F06410 + 1;
    v5 = dword_4F06BA0;
    v13 = a1 > 1;
    v17 = unk_4F06B98;
    v18 = (1LL << ((unsigned __int8)dword_4F06BA0 - 1)) | ((1LL << ((unsigned __int8)dword_4F06BA0 - 1)) - 1);
    v4 = 0;
    if ( a1 > 1 || dword_4F077C4 != 2 )
    {
      v16 = 0;
      v11 = sub_72BA30(5u);
    }
    else
    {
      v13 = 0;
      v11 = sub_72BA30(0);
      v16 = 0;
    }
    goto LABEL_6;
  }
  if ( v3 != 76 )
LABEL_89:
    sub_721090();
  v4 = 1;
  v5 = unk_4F06B88 * dword_4F06BA0;
  v17 = byte_4B6DF90[byte_4F06B90[0]];
  v16 = 0;
  v11 = sub_72C270();
  v22 = qword_4F06410 + 2;
  v13 = a1 > 1;
  v18 = (1LL << ((unsigned __int8)v5 - 1)) | ((1LL << ((unsigned __int8)v5 - 1)) - 1);
LABEL_6:
  v30 = 0;
  v6 = 0;
  v26[1] = 0;
  v26[0] = &v22;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  sub_620DE0(&v23, 0);
  v19 = 0;
  v14 = 0;
  v15 = 0;
  while ( (unsigned __int64)v22 < qword_4F06408 || v27 > 0 )
  {
    if ( v4 == 3 )
    {
      sub_7CD760((__int64)v26, 1, (__int64 *)&v21, v18);
      if ( v6 && dword_4F077C4 == 2 )
      {
        if ( !HIDWORD(qword_4F077B4) )
          goto LABEL_59;
        if ( (_DWORD)qword_4F077B4 )
        {
          v15 = 1;
          goto LABEL_25;
        }
      }
      v7 = sub_722B20(v21, v25);
      if ( v7 == 1 )
      {
        v21 = (unsigned __int16)v25[0];
      }
      else
      {
        v14 = 1;
        if ( v7 > 1 )
          v21 = (unsigned __int16)v25[v7 - 1];
      }
      if ( v6 )
      {
LABEL_25:
        sub_620DE0(&v23, 0);
        goto LABEL_13;
      }
LABEL_32:
      sub_620DE0(v24, v21);
      if ( !unk_4F06B94 )
      {
        if ( v17 )
          sub_6215A0(v24, v5);
        goto LABEL_17;
      }
      if ( v17 )
        sub_6215A0(v24, v5);
      goto LABEL_35;
    }
    if ( v4 == 4 )
    {
      sub_7CD760((__int64)v26, 1, (__int64 *)&v21, v18);
      if ( v6 )
      {
        if ( dword_4F077C4 != 2 )
          goto LABEL_25;
        if ( HIDWORD(qword_4F077B4) )
        {
          if ( (_DWORD)qword_4F077B4 )
            v15 = 1;
          goto LABEL_25;
        }
LABEL_59:
        v15 = 1;
        goto LABEL_25;
      }
      goto LABEL_32;
    }
    if ( v4 != 1 )
    {
      sub_7CD070((__int64)v26, 1, (__int64 *)&v21, v18, 1, v16);
      if ( *(_QWORD *)&dword_4F06B20 <= v6 && !HIDWORD(qword_4F077B4) )
      {
        v15 = 1;
        goto LABEL_11;
      }
      if ( !v6 )
        goto LABEL_32;
      if ( v16 )
      {
        v15 = 1;
      }
      else
      {
LABEL_11:
        if ( !v6 )
          goto LABEL_32;
      }
      if ( (unsigned __int8)(v4 - 3) <= 1u )
        goto LABEL_25;
LABEL_13:
      sub_620DE0(v24, v21);
      if ( !unk_4F06B94 )
      {
        if ( v17 )
        {
          sub_6215A0(v24, v5);
          sub_621EE0(v25, v19);
          sub_6213D0((__int64)&v23, (__int64)v25);
        }
        sub_621410((__int64)v24, v19, &v20);
        goto LABEL_17;
      }
LABEL_35:
      sub_621410((__int64)&v23, v5, &v20);
LABEL_17:
      sub_6213B0((__int64)&v23, (__int64)v24);
      goto LABEL_18;
    }
    sub_7CD760((__int64)v26, 1, (__int64 *)&v21, v18);
    if ( !v6 )
      goto LABEL_32;
LABEL_18:
    v19 += v5;
    ++v6;
  }
  if ( v6 == 1 && v13 )
  {
    if ( v4 )
    {
      if ( !v14 )
        goto LABEL_76;
    }
    else
    {
      if ( dword_4F077C4 != 2 )
      {
        if ( !v14 )
          goto LABEL_76;
LABEL_54:
        sub_7B0EB0((unsigned __int64)qword_4F06410, (__int64)dword_4F07508);
        sub_684B30(0xCB7u, dword_4F07508);
        goto LABEL_55;
      }
      v11 = sub_72BA30(0);
      if ( !v14 )
      {
LABEL_76:
        if ( v15 )
        {
LABEL_77:
          *a2 = v16 == 0 ? 26 : 2688;
          *a3 = qword_4F06410;
          sub_72C970((__int64)xmmword_4F06300);
          goto LABEL_56;
        }
LABEL_55:
        *a2 = 0;
        *a3 = 0;
        goto LABEL_56;
      }
    }
LABEL_53:
    if ( dword_4F077C4 == 2 )
    {
      *a2 = 1535;
      *a3 = qword_4F06410 + 2;
      sub_72C970((__int64)xmmword_4F06300);
LABEL_56:
      result = a2;
      if ( !*a2 )
        goto LABEL_71;
      return result;
    }
    goto LABEL_54;
  }
  if ( v14 )
    goto LABEL_53;
  if ( v15 )
    goto LABEL_77;
  *a2 = 0;
  *a3 = 0;
  if ( a1 <= 1 )
    goto LABEL_56;
  v9 = v4 == 0 ? 1422 : 26;
  if ( HIDWORD(qword_4F077B4) && v6 > *(_QWORD *)&dword_4F06B20 )
  {
    sub_621EE0(v25, dword_4F06B20 * dword_4F06BA0);
    v9 = 1654;
    sub_6213D0((__int64)&v23, (__int64)v25);
  }
  sub_7B0EB0((unsigned __int64)qword_4F06410, (__int64)dword_4F07508);
  sub_684B30(v9, dword_4F07508);
  result = a2;
  if ( !*a2 )
  {
LABEL_71:
    sub_724C70((__int64)xmmword_4F06300, 1);
    v10 = _mm_loadu_si128(&v23);
    xmmword_4F06380[0].m128i_i64[0] = (__int64)v11;
    *(__m128i *)word_4F063B0 = v10;
    result = (_DWORD *)(unk_4F063A8 & 0xF8);
    unk_4F063A8 = unk_4F063A8 & 0xF8 | v4;
  }
  return result;
}
