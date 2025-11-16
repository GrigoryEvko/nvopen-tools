// Function: sub_7B6B00
// Address: 0x7b6b00
//
__int64 __fastcall sub_7B6B00(_QWORD *a1, int a2, char a3, char a4, const char *a5, int a6, _BYTE *a7, __int64 **a8)
{
  _BYTE *v8; // r12
  _BYTE *v9; // rax
  __int64 *v10; // r14
  char v11; // bl
  int v12; // r11d
  __int64 v13; // r13
  char v14; // bl
  _BYTE *v15; // rdx
  int v16; // eax
  __int64 v17; // rdi
  _BYTE *v18; // rdx
  __int64 *v19; // rax
  __int64 v20; // r8
  __int64 *v21; // rax
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // r9
  _QWORD *i; // rdx
  _BYTE *v25; // rsi
  int v26; // esi
  __int64 v27; // rbx
  int v28; // eax
  int v29; // edx
  unsigned int v30; // eax
  _BYTE *v31; // rax
  _BYTE *v32; // r13
  int v33; // edi
  __int64 v34; // rax
  int v35; // eax
  int v36; // eax
  int v37; // eax
  bool v38; // zf
  int v39; // eax
  __int64 *v40; // rax
  int v41; // esi
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // r8
  _BOOL4 v44; // r9d
  _BYTE *v46; // rax
  int v47; // ecx
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 *v50; // rdx
  int v51; // [rsp+4h] [rbp-9Ch]
  bool v53; // [rsp+17h] [rbp-89h]
  _BYTE *v54; // [rsp+20h] [rbp-80h]
  const char *v55; // [rsp+30h] [rbp-70h]
  int v56; // [rsp+30h] [rbp-70h]
  int v57; // [rsp+38h] [rbp-68h]
  int v58; // [rsp+38h] [rbp-68h]
  int v59; // [rsp+38h] [rbp-68h]
  __int64 v60; // [rsp+38h] [rbp-68h]
  int v61; // [rsp+38h] [rbp-68h]
  int v62; // [rsp+38h] [rbp-68h]
  int v63; // [rsp+38h] [rbp-68h]
  unsigned int v64; // [rsp+40h] [rbp-60h]
  int v65; // [rsp+44h] [rbp-5Ch]
  int v69; // [rsp+58h] [rbp-48h]
  int v71; // [rsp+5Ch] [rbp-44h]
  int v72; // [rsp+64h] [rbp-3Ch] BYREF
  unsigned __int64 v73[7]; // [rsp+68h] [rbp-38h] BYREF
  unsigned __int64 v74; // [rsp+B0h] [rbp+10h]

  v53 = (a3 & 0x10) != 0;
  v8 = qword_4F06460;
  v65 = a3 & 7;
  v9 = qword_4F06460;
  if ( a7 )
    v9 = a7;
  v10 = 0;
  v74 = (unsigned __int64)v9;
  v69 = a3 & 8;
  if ( (a3 & 8) != 0 )
  {
    v10 = (__int64 *)unk_4F06458;
    if ( unk_4F06458 )
    {
      if ( !a8 || (v10 = *a8) != 0 )
      {
        do
        {
          if ( v10[1] >= (unsigned __int64)qword_4F06460 )
            break;
          v10 = (__int64 *)*v10;
        }
        while ( v10 );
      }
    }
  }
  v11 = *qword_4F06460;
  v51 = 0;
  v12 = 0;
  v13 = 0;
  v64 = 1194 - (a2 == 0);
LABEL_11:
  while ( 2 )
  {
    while ( 2 )
    {
      if ( a4 != v11 )
      {
        while ( 1 )
        {
          if ( v11 != 92 || a2 )
            goto LABEL_40;
          if ( v69 )
          {
            if ( !v10 || (_BYTE *)v10[1] != v8 )
              goto LABEL_57;
LABEL_75:
            v30 = *((_DWORD *)v10 + 4);
            if ( v30 != 2 )
            {
              if ( v30 <= 2 )
              {
                if ( v30 )
                {
                  v13 += 2;
                }
                else
                {
                  v13 += 3;
                  qword_4F06460 = v8 + 1;
                  v11 = *++v8;
                }
                goto LABEL_79;
              }
              if ( v30 != 3 )
LABEL_153:
                sub_721090();
            }
            ++v13;
            qword_4F06460 = v8 + 2;
            v11 = v8[2];
            v8 += 2;
LABEL_79:
            v10 = (__int64 *)*v10;
            goto LABEL_11;
          }
          qword_4F06460 = v8 + 1;
          v14 = v8[1];
          if ( !v14 )
          {
            ++v8;
            goto LABEL_137;
          }
          if ( (v14 & 0xDF) == 0x55 )
            break;
          v15 = v8 + 2;
          ++v13;
          qword_4F06460 = v8 + 2;
          if ( (unsigned __int8)(v14 - 48) > 7u )
          {
            if ( v14 == 120 )
            {
              v60 = v13;
              v32 = v8 + 2;
              v56 = v12;
              while ( 1 )
              {
                v33 = (unsigned __int8)*v32;
                v8 = v32++;
                v11 = v33;
                if ( !isxdigit(v33) )
                  break;
                qword_4F06460 = v32;
              }
              v13 = v60;
              v12 = v56;
              goto LABEL_11;
            }
LABEL_19:
            v11 = v8[2];
            goto LABEL_20;
          }
          v11 = v8[2];
          if ( (unsigned __int8)(v11 - 48) <= 7u )
          {
            v31 = v8 + 3;
            qword_4F06460 = v8 + 3;
            v11 = v8[3];
            if ( (unsigned __int8)(v11 - 48) <= 7u )
            {
              qword_4F06460 = v8 + 4;
              v11 = v8[4];
              v8 += 4;
              goto LABEL_11;
            }
            goto LABEL_84;
          }
LABEL_20:
          v8 = v15;
          if ( a4 == v11 )
            goto LABEL_21;
        }
        if ( unk_4D042A0 )
        {
          v61 = v12;
          qword_4F06460 = v8;
          sub_7B39D0((unsigned __int64 *)&qword_4F06460, 0, 0, 0);
          v12 = v61;
          if ( v14 == 85 && v53 && ((v65 - 3) & 0xFFFFFFFD) == 0 )
          {
            v8 = qword_4F06460;
            v13 += 2;
            v11 = *qword_4F06460;
          }
          else
          {
            v8 = qword_4F06460;
            v34 = v13 + 4;
            ++v13;
            v11 = *qword_4F06460;
            if ( (unsigned int)(v65 - 1) <= 1 )
              v13 = v34;
          }
          continue;
        }
        v15 = v8 + 2;
        ++v13;
        qword_4F06460 = v8 + 2;
        goto LABEL_19;
      }
      break;
    }
LABEL_21:
    if ( !v69 )
    {
      qword_4F06408 = v8;
      if ( unk_4D041A0 && dword_4D0432C && unk_4F064A8 )
        goto LABEL_120;
      goto LABEL_150;
    }
    v16 = a6 + 2;
    if ( a5[a6] != 91 )
      v16 = a6;
    v17 = v16;
    if ( v16 + (__int64)a6 + 1 >= v8 - a5 )
      goto LABEL_74;
    v18 = &v8[~(__int64)v16];
    v12 = *v18 == 41;
    if ( *v18 != 41 )
    {
      v19 = (__int64 *)unk_4F06458;
      if ( unk_4F06458 )
      {
        if ( *v18 == 93 )
        {
          while ( 1 )
          {
            v25 = (_BYTE *)v19[1];
            if ( v18 <= v25 )
              break;
            v19 = (__int64 *)*v19;
            if ( !v19 )
              goto LABEL_28;
          }
          while ( v18 == v25 )
          {
            v26 = *((_DWORD *)v19 + 4);
            v19 = (__int64 *)*v19;
            if ( !v26 )
              v12 = 1;
            if ( !v19 )
              break;
            v25 = (_BYTE *)v19[1];
          }
        }
      }
    }
LABEL_28:
    v57 = v17;
    v55 = &a5[a6];
    if ( !v12 )
    {
LABEL_40:
      if ( v10 && (_BYTE *)v10[1] == v8 )
        goto LABEL_75;
      if ( !v11 )
      {
        if ( v8[1] == 6 )
        {
          ++v13;
          if ( a6 < 0 )
          {
            v59 = v12;
            if ( unk_4D04328 )
            {
              sub_7B0EB0((unsigned __int64)v8, (__int64)dword_4F07508);
              sub_684AC0(5u, v64);
            }
            else
            {
              sub_7B0EB0((unsigned __int64)v8, (__int64)dword_4F07508);
              sub_684AC0(7u, 0x35Du);
            }
            v8 = qword_4F06460;
            v12 = v59;
          }
          qword_4F06460 = v8 + 2;
          v11 = v8[2];
          v8 += 2;
          continue;
        }
LABEL_137:
        qword_4F06408 = v8;
        if ( unk_4D041A0 )
        {
          if ( dword_4D0432C )
          {
            if ( unk_4F064A8 )
            {
              v40 = (__int64 *)qword_4F084D0;
              if ( qword_4F084D0 )
              {
                v41 = 1;
                goto LABEL_142;
              }
            }
          }
        }
LABEL_147:
        qword_4F06408 = v8 - 1;
        *a1 += v13;
        return 1;
      }
LABEL_57:
      if ( !dword_4D0432C )
      {
        v31 = v8 + 1;
        qword_4F06460 = v8 + 1;
        if ( v65 == 2 )
        {
          v62 = v12;
          v35 = sub_722A20(v11, v73);
          v8 = qword_4F06460;
          v12 = v62;
          v11 = *qword_4F06460;
          v13 += v35;
          continue;
        }
        v11 = v8[1];
        ++v13;
LABEL_84:
        v8 = v31;
        continue;
      }
      v72 = 0;
      if ( unk_4D041A0 && unk_4F064A8 )
      {
        v63 = v12;
        v36 = sub_722680(v8, v73, &v72, 0);
        v12 = v63;
        v27 = v36;
        if ( !v72 && v73[0] > 0x2000 )
        {
          v37 = sub_7AB890(v73[0], 1u);
          v12 = v63;
          v38 = v37 == 0;
          v39 = 1;
          if ( v38 )
            v39 = v51;
          v51 = v39;
        }
      }
      else
      {
        if ( (char)*v8 >= 0 )
        {
          v27 = 1;
          qword_4F06460 = v8 + 1;
          goto LABEL_62;
        }
        v58 = v12;
        v28 = sub_721AB0(v8, &v72, unk_4F064A8 == 0);
        v12 = v58;
        v27 = v28;
      }
      if ( (int)v27 > 1
        && qword_4F06498 <= (unsigned __int64)qword_4F06460
        && unk_4F06490 > (unsigned __int64)qword_4F06460 )
      {
        v46 = qword_4F06460 + 1;
        v47 = 1;
        while ( 1 )
        {
          ++v47;
          qword_4F06460 = v46 + 1;
          v48 = *(int *)&word_4F06480;
          ++*(_DWORD *)&word_4F06480;
          qword_4F06488[v48] = v46;
          if ( (_DWORD)v27 == v47 )
            break;
          v46 = qword_4F06460;
        }
        v29 = v72;
      }
      else
      {
        v29 = v72;
        qword_4F06460 += (int)v27;
      }
      if ( v29 )
      {
LABEL_71:
        ++v13;
LABEL_64:
        v8 = qword_4F06460;
        v11 = *qword_4F06460;
        continue;
      }
LABEL_62:
      switch ( v65 )
      {
        case 1:
          v13 += v27;
          goto LABEL_64;
        case 2:
          v13 += 4;
          goto LABEL_64;
        case 3:
        case 5:
          v13 += 2;
          goto LABEL_64;
        case 4:
          goto LABEL_71;
        default:
          goto LABEL_153;
      }
    }
    break;
  }
  v54 = &v8[-v17];
  v12 = strncmp(a5, &v8[-v17], a6);
  if ( v12 )
  {
LABEL_74:
    v12 = 0;
    goto LABEL_40;
  }
  v20 = a6;
  v21 = (__int64 *)unk_4F06458;
  if ( !unk_4F06458 )
    goto LABEL_113;
  while ( 1 )
  {
    v22 = v21[1];
    if ( (unsigned __int64)a5 <= v22 )
      break;
    v21 = (__int64 *)*v21;
    if ( !v21 )
      goto LABEL_113;
  }
  v23 = v21[1];
  for ( i = v21; ; v23 = i[1] )
  {
    if ( (unsigned __int64)v54 <= v23 )
    {
      if ( (unsigned __int64)v55 <= v22 )
      {
        if ( i[1] >= (unsigned __int64)v8 )
          goto LABEL_113;
      }
      else if ( i[1] < (unsigned __int64)v8 )
      {
        while ( 1 )
        {
          v42 = v21[1];
          if ( (unsigned __int64)v55 <= v42 )
            break;
          if ( i )
          {
            v43 = i[1];
            if ( v43 < (unsigned __int64)v8 )
            {
              v44 = *((_DWORD *)v21 + 4) != 3;
              v21 = (__int64 *)*v21;
              i = (_QWORD *)*i;
              v12 += 3 * v44 - 1;
              if ( v42 - (_QWORD)a5 == v43 - (_QWORD)v54 )
                continue;
            }
          }
          goto LABEL_40;
        }
        v20 = a6;
        if ( !i || (unsigned __int64)v8 <= i[1] )
        {
          if ( a6 != v57 )
          {
            v12 += 2;
            goto LABEL_115;
          }
          goto LABEL_117;
        }
      }
      goto LABEL_40;
    }
    i = (_QWORD *)*i;
    if ( !i )
      break;
  }
  if ( (unsigned __int64)v55 > v22 )
    goto LABEL_40;
LABEL_113:
  if ( a6 != v57 )
  {
    v12 = 2;
LABEL_115:
    if ( *(v8 - 2) != 63 || *(v8 - 1) != 63 )
      goto LABEL_40;
  }
LABEL_117:
  qword_4F06408 = v8;
  if ( unk_4D041A0 && dword_4D0432C && unk_4F064A8 )
  {
LABEL_120:
    if ( qword_4F084D0 || (v51 & 1) != 0 )
    {
      v71 = v12;
      sub_7B0EB0(v74, (__int64)dword_4F07508);
      sub_684AC0(5u, 0xC9Du);
      v40 = (__int64 *)qword_4F084D0;
      v12 = v71;
      if ( qword_4F084D0 )
      {
        v41 = 0;
LABEL_142:
        v49 = qword_4F084C8;
        while ( 1 )
        {
          v50 = (__int64 *)*v40;
          *v40 = v49;
          v49 = (__int64)v40;
          if ( !v50 )
            break;
          v40 = v50;
        }
        qword_4F084C8 = (__int64)v40;
        qword_4F084D0 = 0;
        if ( v41 )
        {
          v8 = (_BYTE *)qword_4F06408;
          goto LABEL_147;
        }
      }
    }
    if ( v69 )
    {
      v20 = a6;
      goto LABEL_131;
    }
LABEL_150:
    *a1 += v13;
    return 0;
  }
  else
  {
LABEL_131:
    *a1 = v13 + *a1 - 1 - v20 - v12;
    return 0;
  }
}
