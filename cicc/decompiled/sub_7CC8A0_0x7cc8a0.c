// Function: sub_7CC8A0
// Address: 0x7cc8a0
//
__int64 __fastcall sub_7CC8A0(int a1, _DWORD *a2, _QWORD *a3, unsigned __int8 *a4)
{
  _DWORD *v4; // rbx
  const char *v5; // rdx
  char v6; // r12
  const char *v7; // rax
  char v8; // dl
  const char *v9; // rdx
  unsigned __int8 v10; // r14
  char v11; // al
  const char *v12; // r15
  const char *v13; // rdx
  const char *v14; // r9
  __int64 result; // rax
  int v16; // edx
  __int64 v17; // rdi
  _QWORD *v18; // rdi
  unsigned __int64 v19; // rdx
  __int64 v20; // rax
  const char *v21; // r12
  unsigned __int64 v22; // rbx
  char v23; // cl
  _QWORD *v24; // rax
  __m128i v25; // xmm1
  char v26; // dl
  __int64 v27; // rax
  const char *v28; // rdx
  int v29; // ecx
  unsigned __int8 v30; // cl
  const char *v31; // [rsp+8h] [rbp-88h]
  const char *v32; // [rsp+8h] [rbp-88h]
  unsigned __int64 v33; // [rsp+10h] [rbp-80h]
  char v34; // [rsp+10h] [rbp-80h]
  const char *v35; // [rsp+10h] [rbp-80h]
  int v39; // [rsp+34h] [rbp-5Ch]
  char v40; // [rsp+38h] [rbp-58h]
  int v41; // [rsp+40h] [rbp-50h] BYREF
  int v42; // [rsp+44h] [rbp-4Ch] BYREF
  int v43; // [rsp+48h] [rbp-48h] BYREF
  __m128i v44[4]; // [rsp+50h] [rbp-40h] BYREF

  v4 = a2;
  v42 = 0;
  v5 = (const char *)qword_4F06408;
  *a2 = 0;
  v6 = *v5;
  if ( (unsigned __int8)((*v5 & 0xDF) - 73) > 1u )
  {
    v39 = 0;
    v7 = v5;
  }
  else
  {
    v39 = 1;
    v6 = *(v5 - 1);
    v7 = v5 - 1;
  }
  v8 = v6 & 0xDF;
  if ( (v6 & 0xDF) == 0x46 )
  {
    v9 = v7 - 1;
    v10 = 2;
    v40 = *(v7 - 1);
    goto LABEL_5;
  }
  if ( v8 == 76 )
  {
    v9 = v7 - 1;
    v10 = 6;
    v40 = *(v7 - 1);
    if ( (unsigned __int8)((v40 & 0xDF) - 73) <= 1u )
      goto LABEL_6;
LABEL_27:
    v11 = v40;
    v12 = v9;
    v40 = v6;
    v6 = v9[2];
    goto LABEL_7;
  }
  if ( dword_4D04288 && v8 == 87 )
  {
    v9 = v7 - 1;
    v40 = *(v7 - 1);
    v10 = unk_4B6D467;
    goto LABEL_5;
  }
  if ( dword_4D04284 && v8 == 81 )
  {
    v9 = v7 - 1;
    v40 = *(v7 - 1);
    v10 = unk_4B6D466;
    goto LABEL_5;
  }
  v10 = dword_4D04190;
  if ( !dword_4D04190 )
  {
    if ( dword_4F077C0 )
    {
      if ( (_DWORD)qword_4F077B4 || qword_4F077A8 <= 0x1116Fu )
        goto LABEL_55;
    }
    else if ( !dword_4F077BC || (_DWORD)qword_4F077B4 || qword_4F077A8 <= 0x1FBCFu )
    {
LABEL_55:
      if ( !unk_4D0428C )
      {
        v40 = v6;
        v9 = v7;
        v10 = 4;
        v6 = v7[1];
        goto LABEL_5;
      }
      if ( v7 > qword_4F06410 + 2 && (*(v7 - 2) & 0xDF) == 0x46 && *(v7 - 1) == 49 && v6 == 54 )
      {
        v9 = v7 - 3;
        v6 = *(v7 - 2);
        v40 = *(v7 - 3);
        goto LABEL_5;
      }
LABEL_69:
      v40 = v6;
      v9 = v7;
      v6 = v7[1];
      v10 = 4;
      goto LABEL_5;
    }
  }
  if ( v7 > qword_4F06410 + 3 && (*(v7 - 3) & 0xDF) == 0x46 )
  {
    v28 = v7 - 2;
    if ( *(v7 - 2) == 49 && v28[1] == 50 && *v7 == 56 )
    {
      v9 = v7 - 4;
      if ( dword_4D04284 )
      {
        v6 = *(v7 - 3);
        v10 = 13;
        v40 = *(v7 - 4);
      }
      else
      {
        *a2 = 3282;
        *a3 = v7 - 7;
        v30 = 5;
        if ( dword_4D04964 )
          v30 = byte_4F07472[0];
        v10 = 12;
        *a4 = v30;
        v6 = *(v7 - 3);
        v40 = *(v7 - 4);
      }
    }
    else
    {
      v29 = *(unsigned __int8 *)v28;
      if ( v29 == 51 && v28[1] == 50 && *v7 == 120 )
      {
        v9 = v7 - 4;
        v6 = *(v7 - 3);
        v10 = 3;
        v40 = *(v7 - 4);
      }
      else
      {
        if ( v29 != 54 || v28[1] != 52 || *v7 != 120 )
        {
          if ( !a1 )
            sub_721090();
          goto LABEL_69;
        }
        v9 = v7 - 4;
        v6 = *(v7 - 3);
        v10 = 5;
        v40 = *(v7 - 4);
      }
    }
  }
  else
  {
    if ( v7 <= qword_4F06410 + 2 || (*(v7 - 2) & 0xDF) != 0x46 )
      goto LABEL_69;
    v26 = *(v7 - 1);
    v40 = *(v7 - 3);
    if ( (v40 & 0xDF) == 0x42 && v26 == 49 && v6 == 54 )
    {
      v9 = v7 - 4;
      v6 = *(v7 - 3);
      v10 = 9;
      v40 = *(v7 - 4);
    }
    else if ( v26 == 51 )
    {
      if ( v6 != 50 )
        goto LABEL_69;
      v9 = v7 - 3;
      v6 = *(v7 - 2);
      v10 = 11;
    }
    else if ( v26 == 54 )
    {
      if ( v6 != 52 )
        goto LABEL_69;
      v9 = v7 - 3;
      v6 = *(v7 - 2);
      v10 = 12;
    }
    else
    {
      if ( v26 != 49 || v6 != 54 )
        goto LABEL_69;
      v10 = 0;
      if ( HIDWORD(qword_4F077B4) )
        v10 = qword_4F077A8 >= 0x1FBD0u ? 0xA : 0;
      v9 = v7 - 3;
      v6 = *(v7 - 2);
    }
  }
LABEL_5:
  if ( (unsigned __int8)((v40 & 0xDF) - 73) > 1u )
    goto LABEL_27;
LABEL_6:
  v39 = 1;
  v11 = *(v9 - 1);
  v12 = v9 - 1;
LABEL_7:
  if ( (v11 & 0xDF) == 0x45 || ((v11 - 43) & 0xFD) == 0 && qword_4F06410 != v12 && (*(v12 - 1) & 0xDF) == 0x45 )
  {
    v13 = v12 + 1;
    *(_WORD *)(v12 + 1) = 48;
  }
  else
  {
    *((_BYTE *)v12 + 1) = 0;
    v13 = v12;
  }
  v14 = qword_4F06410;
  if ( unk_4F061E0 )
  {
    v17 = qword_4F17FE8;
    if ( !qword_4F17FE8 )
    {
      v32 = qword_4F06410;
      v35 = v13;
      v27 = sub_8237A0(64);
      v14 = v32;
      v13 = v35;
      qword_4F17FE8 = v27;
      v17 = v27;
    }
    v31 = v14;
    v33 = (unsigned __int64)v13;
    sub_823800(v17);
    v18 = (_QWORD *)qword_4F17FE8;
    v19 = v33;
    v20 = *(_QWORD *)(qword_4F17FE8 + 16);
    if ( v33 >= (unsigned __int64)v31 )
    {
      v34 = v6;
      v21 = v31;
      v22 = v19;
      do
      {
        v23 = *v21;
        if ( *v21 != 39 )
        {
          if ( (unsigned __int64)(v20 + 1) > v18[1] )
          {
            sub_823810(v18);
            v18 = (_QWORD *)qword_4F17FE8;
            v23 = *v21;
            v20 = *(_QWORD *)(qword_4F17FE8 + 16);
          }
          *(_BYTE *)(v18[4] + v20) = v23;
          v20 = v18[2] + 1LL;
          v18[2] = v20;
        }
        ++v21;
      }
      while ( v22 >= (unsigned __int64)v21 );
      v6 = v34;
      v4 = a2;
    }
    if ( (unsigned __int64)(v20 + 1) > v18[1] )
    {
      sub_823810(v18);
      v18 = (_QWORD *)qword_4F17FE8;
      v20 = *(_QWORD *)(qword_4F17FE8 + 16);
    }
    *(_BYTE *)(v18[4] + v20) = 0;
    v14 = (const char *)v18[4];
    ++v18[2];
  }
  if ( a1 )
    sub_70AF10(v10, (__int64)v14, v44, &v41, &v42);
  else
    sub_70AFD0(v10, v14, v44, &v41);
  *((_BYTE *)v12 + 2) = v6;
  *((_BYTE *)v12 + 1) = v40;
  if ( v41 )
  {
    *v4 = 30;
    *a3 = qword_4F06410;
    result = (__int64)a4;
    *a4 = 8;
    if ( !*v4 )
      return result;
    return sub_72C970((__int64)xmmword_4F06300);
  }
  if ( !v39 )
  {
    sub_724C70((__int64)xmmword_4F06300, 3);
    v24 = sub_72C610(v10);
    v25 = _mm_loadu_si128(v44);
    xmmword_4F06380[0].m128i_i64[0] = (__int64)v24;
    *(__m128i *)word_4F063B0 = v25;
    if ( !v42 )
      goto LABEL_22;
LABEL_42:
    sub_7B0EB0((unsigned __int64)qword_4F06410, (__int64)&v43);
    sub_684B30(0x416u, &v43);
    goto LABEL_22;
  }
  sub_724C70((__int64)xmmword_4F06300, 4);
  xmmword_4F06380[0].m128i_i64[0] = (__int64)sub_72C6F0(v10);
  sub_70B680(v10, 0, *(_OWORD **)word_4F063B0, &v41);
  v16 = v42;
  *(__m128i *)(*(_QWORD *)word_4F063B0 + 16LL) = _mm_loadu_si128(v44);
  if ( v16 )
    goto LABEL_42;
LABEL_22:
  result = (unsigned int)*v4;
  if ( (_DWORD)result )
  {
    result = (__int64)a4;
    if ( *a4 > 5u )
      return sub_72C970((__int64)xmmword_4F06300);
  }
  return result;
}
