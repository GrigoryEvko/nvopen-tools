// Function: sub_7A7550
// Address: 0x7a7550
//
__int64 __fastcall sub_7A7550(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  __int64 v4; // rdi
  __int64 v5; // rcx
  unsigned __int16 v6; // ax
  int v7; // r15d
  int v8; // edx
  int v9; // r8d
  _BOOL4 v10; // r13d
  unsigned int v11; // r13d
  __int64 v12; // rax
  __int64 result; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  int v16; // r9d
  unsigned __int64 v17; // rax
  bool v18; // cf
  bool v19; // zf
  __int64 v20; // rax
  __int64 v21; // r9
  __int64 v22; // rdi
  __int64 v23; // rsi
  int v24; // eax
  __int64 v25; // rcx
  _DWORD *v26; // rdx
  _BYTE *v27; // rax
  int v28; // r13d
  __int64 v29; // rax
  __int16 v30; // ax
  __int64 v31; // rcx
  char *v32; // rdi
  _BYTE *v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rsi
  __int64 v38; // rdi
  __int64 v39; // rcx
  unsigned __int8 v40; // r9
  __int64 *v41; // rdx
  __int64 *v42; // r15
  int v43; // eax
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // rax
  int v47; // r13d
  __int64 v48; // rdx
  unsigned __int8 v49; // [rsp+Ch] [rbp-74h]
  int v50; // [rsp+Ch] [rbp-74h]
  __int64 *v51; // [rsp+10h] [rbp-70h]
  unsigned __int8 v52; // [rsp+10h] [rbp-70h]
  unsigned __int8 v53; // [rsp+10h] [rbp-70h]
  unsigned __int8 v54; // [rsp+10h] [rbp-70h]
  char v55; // [rsp+10h] [rbp-70h]
  int v56; // [rsp+10h] [rbp-70h]
  unsigned __int8 s1; // [rsp+18h] [rbp-68h]
  char *s1a; // [rsp+18h] [rbp-68h]
  char *s1b; // [rsp+18h] [rbp-68h]
  char s1c; // [rsp+18h] [rbp-68h]
  char s1d; // [rsp+18h] [rbp-68h]
  unsigned int v62; // [rsp+2Ch] [rbp-54h] BYREF
  char s[80]; // [rsp+30h] [rbp-50h] BYREF

  v62 = 0;
  sub_7C9660(a1);
  v2 = qword_4F061C8;
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  ++*(_BYTE *)(v2 + 9);
  ++*(_BYTE *)(v2 + 12);
  if ( HIDWORD(qword_4F077B4) && word_4F06418[0] != 27 )
  {
    v3 = (__int64)dword_4F07508;
    v4 = 125;
    sub_684B30(0x7Du, dword_4F07508);
    v6 = word_4F06418[0];
    if ( word_4F06418[0] != 1 )
      goto LABEL_4;
  }
  else
  {
    v3 = 125;
    v4 = 27;
    sub_7BE280(27, 125, 0, 0);
    v6 = word_4F06418[0];
    if ( word_4F06418[0] != 1 )
    {
LABEL_4:
      v7 = 0;
      v8 = 0;
      v9 = 0;
      v10 = 0;
      goto LABEL_5;
    }
  }
  v15 = qword_4D04A00;
  v16 = HIDWORD(qword_4F077B4) == 0 ? 8 : 5;
  v17 = *(_QWORD *)(qword_4D04A00 + 16);
  v18 = v17 < 4;
  v19 = v17 == 4;
  if ( v17 != 4 )
  {
    v7 = 0;
    if ( v17 != 3 )
      goto LABEL_23;
    v27 = *(_BYTE **)(qword_4D04A00 + 8);
    if ( *v27 == 112 && v27[1] == 111 )
    {
      v28 = (unsigned __int8)v27[2] - 112;
      if ( v27[2] == 112 )
      {
        if ( !qword_4F083D0 )
        {
          v4 = (unsigned __int8)(HIDWORD(qword_4F077B4) == 0 ? 8 : 5);
          v3 = 689;
          s1d = HIDWORD(qword_4F077B4) == 0 ? 8 : 5;
          sub_684AC0(v4, 0x2B1u);
          LOBYTE(v16) = s1d;
        }
        s1 = v16;
        sub_7B8B50(v4, v3, v15, v5);
        if ( word_4F06418[0] == 28 )
        {
          v29 = qword_4F083D0;
          v9 = 1;
          if ( qword_4F083D0 )
          {
LABEL_40:
            v7 = 1;
            v10 = 0;
            dword_4D03BE8[0] = *(_DWORD *)(v29 + 16);
            qword_4F083D0 = *(_QWORD *)v29;
            *(_QWORD *)v29 = qword_4F083C8;
            v8 = 0;
            qword_4F083C8 = v29;
            v6 = word_4F06418[0];
            goto LABEL_5;
          }
          goto LABEL_73;
        }
        v37 = 253;
        v38 = 67;
        sub_7BE280(67, 253, 0, 0);
        v40 = s1;
        if ( word_4F06418[0] != 1 )
        {
          s1b = 0;
          v9 = 1;
          v42 = 0;
          goto LABEL_70;
        }
        v41 = (__int64 *)qword_4F083D0;
        v42 = 0;
        s1b = *(char **)(qword_4D04A00 + 8);
        if ( qword_4F083D0 )
        {
          v42 = (__int64 *)qword_4F083D0;
          if ( s1b )
          {
            while ( 1 )
            {
              v37 = v42[1];
              if ( v37 )
              {
                v38 = (__int64)s1b;
                v49 = v40;
                v51 = v41;
                v43 = strcmp(s1b, (const char *)v37);
                v41 = v51;
                v40 = v49;
                if ( !v43 )
                  break;
              }
              v42 = (__int64 *)*v42;
              if ( !v42 )
              {
                v38 = v40;
                v37 = 688;
                v52 = v40;
                sub_6849F0(v40, 0x2B0u, &dword_4F063F8, (__int64)s1b);
                v40 = v52;
                goto LABEL_68;
              }
            }
            if ( v51 != v42 )
            {
              v39 = qword_4F083C8;
              while ( 1 )
              {
                qword_4F083D0 = *v41;
                *v41 = v39;
                v39 = (__int64)v41;
                qword_4F083C8 = (__int64)v41;
                if ( (__int64 *)qword_4F083D0 == v42 )
                  break;
                v41 = (__int64 *)qword_4F083D0;
              }
            }
          }
        }
LABEL_68:
        v53 = v40;
        sub_7B8B50(v38, v37, v41, v39);
        v40 = v53;
        v9 = 1;
        if ( word_4F06418[0] == 28 )
        {
LABEL_70:
          v29 = qword_4F083D0;
          if ( qword_4F083D0 && (v42 != 0 || s1b == 0 || v40 != 8) )
            goto LABEL_40;
LABEL_73:
          v6 = word_4F06418[0];
          v7 = 0;
          v8 = 0;
          v10 = 0;
          goto LABEL_5;
        }
        goto LABEL_69;
      }
    }
    goto LABEL_51;
  }
  v31 = 4;
  v32 = "push";
  v33 = *(_BYTE **)(qword_4D04A00 + 8);
  do
  {
    if ( !v31 )
      break;
    v18 = *v33 < (unsigned __int8)*v32;
    v19 = *v33++ == (unsigned __int8)*v32++;
    --v31;
  }
  while ( v19 );
  if ( (!v18 && !v19) == v18 )
  {
    s1c = HIDWORD(qword_4F077B4) == 0 ? 8 : 5;
    sub_7B8B50(v32, v33, qword_4D04A00, v31);
    v9 = 0;
    if ( word_4F06418[0] != 28 && (sub_7BE280(67, 253, 0, 0), v9 = 0, word_4F06418[0] == 1) )
    {
      v55 = s1c;
      s1b = *(char **)(qword_4D04A00 + 8);
      sub_7B8B50(67, 253, v44, v45);
      v40 = v55;
      v9 = 0;
      if ( word_4F06418[0] != 28 )
      {
        v28 = 1;
        v42 = 0;
LABEL_69:
        v50 = v9;
        v54 = v40;
        sub_7BE280(67, 253, 0, 0);
        v40 = v54;
        v9 = v50;
        if ( !v28 )
          goto LABEL_70;
      }
    }
    else
    {
      s1b = 0;
    }
    v46 = qword_4F083C8;
    v47 = dword_4D03BE8[0];
    if ( qword_4F083C8 )
    {
      qword_4F083C8 = *(_QWORD *)qword_4F083C8;
    }
    else
    {
      v56 = v9;
      v46 = sub_823970(24);
      v9 = v56;
    }
    v48 = qword_4F083D0;
    *(_DWORD *)(v46 + 16) = v47;
    v7 = 1;
    v10 = 1;
    *(_QWORD *)v46 = v48;
    v8 = 0;
    *(_QWORD *)(v46 + 8) = s1b;
    qword_4F083D0 = v46;
    v6 = word_4F06418[0];
    goto LABEL_5;
  }
  v19 = memcmp(*(const void **)(qword_4D04A00 + 8), "show", 4u) == 0;
  v10 = !v19;
  if ( !v19 )
  {
LABEL_51:
    v7 = 0;
    goto LABEL_23;
  }
  if ( dword_4D03BE8[0] )
    sprintf(s, "%d", dword_4D03BE8[0]);
  else
    strcpy(s, "not set");
  v7 = 0;
  sub_685190(0x4EAu, (__int64)s);
  sub_7B8B50(1258, s, v35, v36);
  v6 = word_4F06418[0];
  v9 = 0;
  v8 = 1;
LABEL_5:
  if ( v6 == 4 )
  {
    v20 = sub_620FA0((__int64)xmmword_4F06300, &v62);
    v11 = v62;
    v21 = v20;
    if ( HIDWORD(qword_4F077B4) )
    {
      v23 = 0;
      v22 = 0;
      s1a = (char *)v20;
      v11 = 1;
      v30 = sub_7BE840(0, 0);
      v21 = (__int64)s1a;
      if ( v30 != 28 )
        goto LABEL_32;
      v11 = v62;
      v26 = (_DWORD *)HIDWORD(qword_4F077B4);
      if ( HIDWORD(qword_4F077B4) )
      {
        if ( v62 )
        {
LABEL_31:
          v23 = 660;
          v22 = 8;
          v11 = 0;
          sub_684AC0(8u, 0x294u);
          v26 = dword_4D03BE8;
          dword_4D03BE8[0] = 0;
LABEL_32:
          sub_7B8B50(v22, v23, v26, v25);
          goto LABEL_9;
        }
        if ( !s1a )
        {
          v7 = 1;
          dword_4D03BE8[0] = 0;
          goto LABEL_32;
        }
      }
    }
    if ( !v11 )
    {
      v22 = v21;
      v23 = (__int64)dword_4D03BE8;
      v24 = sub_7A7520(v21, dword_4D03BE8);
      v26 = dword_4D03BE8;
      if ( v24 )
      {
        v7 = 1;
        goto LABEL_32;
      }
    }
    goto LABEL_31;
  }
  if ( v6 == 28 )
  {
    v11 = v8 | v10 | v9;
    if ( !v11 )
    {
      v7 = 1;
      dword_4D03BE8[0] = 0;
      goto LABEL_9;
    }
    goto LABEL_8;
  }
  if ( !v8 )
  {
LABEL_23:
    v11 = 0;
    sub_6851D0(0x295u);
    goto LABEL_9;
  }
LABEL_8:
  v11 = 0;
LABEL_9:
  v12 = qword_4F061C8;
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  --*(_BYTE *)(v12 + 9);
  --*(_BYTE *)(v12 + 12);
  if ( !HIDWORD(qword_4F077B4) || word_4F06418[0] == 28 )
  {
    sub_7BE280(28, 18, 0, 0);
  }
  else if ( v11 )
  {
    sub_684B30(0x8FCu, &dword_4F063F8);
  }
  else
  {
    sub_684B30(0x12u, &dword_4F063F8);
  }
  result = sub_7C96B0(v11);
  if ( v7 )
  {
    if ( dword_4F04C58 != -1 )
    {
      result = qword_4F04C68[0] + 776LL * dword_4F04C58;
      if ( (*(_BYTE *)(result + 10) & 2) == 0 )
        return result;
      result = *(_QWORD *)(*(_QWORD *)(result + 184) + 32LL);
      v14 = *(_QWORD *)result;
      goto LABEL_17;
    }
    result = unk_4F04C48;
    if ( unk_4F04C48 != -1 )
    {
      result = 776LL * unk_4F04C48;
      if ( (*(_BYTE *)(qword_4F04C68[0] + result + 10) & 1) != 0 )
      {
        v34 = qword_4F04C68[0] + result + 776;
        result = *(_DWORD *)(qword_4F04C68[0] + result + 780) & 0x200FF;
        if ( (_DWORD)result == 6 )
        {
          result = *(_QWORD *)(*(_QWORD *)(v34 + 184) + 32LL);
          v14 = *(_QWORD *)result;
LABEL_17:
          if ( v14 )
            return sub_685460(0x35Fu, (FILE *)(a1 + 56), v14);
        }
      }
    }
  }
  return result;
}
