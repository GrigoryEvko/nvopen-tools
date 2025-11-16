// Function: sub_817D30
// Address: 0x817d30
//
void __fastcall sub_817D30(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 *a5)
{
  __int64 v7; // rbx
  __int64 v8; // r13
  char v9; // al
  int v10; // r9d
  __int64 v11; // r15
  bool v12; // zf
  char v13; // al
  int v14; // r11d
  __int64 v15; // r10
  __int64 v16; // rax
  __int64 v17; // r10
  __int64 v18; // rdx
  char *v19; // rcx
  char v20; // al
  __int64 v21; // r8
  __int64 v22; // r10
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // r10
  int v26; // esi
  char v27; // si
  char v28; // al
  _BYTE *v29; // rax
  char *v30; // rax
  __int64 *v31; // rax
  __int64 v32; // rax
  char *v33; // rax
  _QWORD *v34; // r14
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  int v38; // [rsp+Ch] [rbp-144h]
  __int64 v39; // [rsp+10h] [rbp-140h]
  __int64 v40; // [rsp+18h] [rbp-138h]
  __int64 *v41; // [rsp+18h] [rbp-138h]
  char *v42; // [rsp+20h] [rbp-130h]
  __int64 v43; // [rsp+28h] [rbp-128h]
  int v44; // [rsp+28h] [rbp-128h]
  __int64 v45; // [rsp+28h] [rbp-128h]
  __int64 v46; // [rsp+30h] [rbp-120h]
  __int64 v47; // [rsp+30h] [rbp-120h]
  int v48; // [rsp+38h] [rbp-118h]
  char v49; // [rsp+38h] [rbp-118h]
  int v50; // [rsp+38h] [rbp-118h]
  int v51; // [rsp+38h] [rbp-118h]
  unsigned int v53; // [rsp+44h] [rbp-10Ch] BYREF
  int v54; // [rsp+48h] [rbp-108h] BYREF
  int v55; // [rsp+4Ch] [rbp-104h] BYREF
  _QWORD v56[10]; // [rsp+50h] [rbp-100h] BYREF
  int v57; // [rsp+A0h] [rbp-B0h]
  int v58; // [rsp+A8h] [rbp-A8h]

  v7 = a2;
  v53 = 0;
  v8 = sub_80AAC0(a1, &v53);
  v9 = *(_BYTE *)(v8 + 24);
  if ( v9 == 2 )
  {
    v11 = *(_QWORD *)(v8 + 64);
    v10 = dword_4D0425C;
    if ( !dword_4D0425C )
      goto LABEL_20;
    v10 = 0;
    if ( !a3 )
      goto LABEL_20;
  }
  else
  {
    v10 = dword_4D0425C;
    if ( v9 == 3 )
    {
      v11 = *(_QWORD *)(v8 + 64);
      if ( !dword_4D0425C || (v10 = 0, !a3) )
      {
LABEL_17:
        a3 = *(_QWORD *)(v8 + 56);
        v17 = 0;
        v18 = 0;
        v19 = 0;
        v20 = *(_BYTE *)(a3 + 89) & 4;
        if ( !v20 )
        {
          v14 = 0;
          v21 = 0;
          a3 = 0;
          goto LABEL_22;
        }
        goto LABEL_18;
      }
    }
    else if ( v9 == 20 )
    {
      v11 = *(_QWORD *)(v8 + 64);
      if ( !dword_4D0425C )
        goto LABEL_89;
      if ( !a3 )
      {
        v14 = 0;
        v10 = 0;
        goto LABEL_57;
      }
    }
    else if ( v9 == 4 )
    {
      v11 = *(_QWORD *)(v8 + 64);
      if ( !dword_4D0425C )
      {
        v15 = *(_QWORD *)(v8 + 56);
        v16 = a3;
        a3 = v15;
        goto LABEL_95;
      }
      if ( !a3 )
      {
        v15 = *(_QWORD *)(v8 + 56);
        v16 = 0;
        v10 = 0;
        a3 = v15;
LABEL_38:
        if ( v11 )
        {
          v23 = *(_QWORD *)(v11 + 8);
          if ( v23 )
          {
            if ( *(_QWORD *)(v23 + 8) == *(_QWORD *)(*(_QWORD *)(v15 + 40) + 32LL) )
            {
              *a5 += 2;
              v11 = 0;
              v48 = v10;
              v43 = v16;
              v46 = v15;
              sub_8238B0(qword_4F18BE0, "sr", 2);
              sub_80F5E0(*(_QWORD *)(*(_QWORD *)(v46 + 40) + 32LL), 0, a5);
              v10 = v48;
              v16 = v43;
              v15 = v46;
            }
          }
        }
LABEL_95:
        if ( !v16 && *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v15 + 40) + 32LL) + 168LL) + 113LL) )
          goto LABEL_43;
        goto LABEL_96;
      }
    }
    else
    {
      if ( v9 == 22 )
      {
        v11 = *(_QWORD *)(v8 + 64);
        if ( !dword_4D0425C )
        {
          v14 = 0;
          v17 = 0;
          v18 = 0;
          v19 = 0;
          v20 = 0;
          v21 = 0;
          a3 = 0;
          goto LABEL_22;
        }
      }
      else
      {
        if ( !dword_4D0425C )
          goto LABEL_43;
        v11 = 0;
      }
      if ( !a3 )
      {
        v17 = 0;
        v18 = 0;
        v19 = 0;
        v21 = 0;
        v10 = 0;
        v20 = 0;
        v14 = 0;
        goto LABEL_22;
      }
    }
  }
  a3 = sub_80AAC0(a3, &v54);
  v12 = (unsigned int)sub_8DBE70(*(_QWORD *)a3) == 0;
  v13 = *(_BYTE *)(v8 + 24);
  v14 = v12;
  if ( v13 == 2 )
  {
    v10 = 1;
LABEL_20:
    v22 = *(_QWORD *)(v8 + 56);
    if ( *(_BYTE *)(v22 + 173) != 12 )
      goto LABEL_21;
    v28 = *(_BYTE *)(v22 + 176);
    v21 = 0;
    if ( v28 == 11 )
    {
      v21 = *(_QWORD *)(v22 + 192);
      v22 = *(_QWORD *)(v22 + 184);
      v28 = *(_BYTE *)(v22 + 176);
    }
    switch ( v28 )
    {
      case 3:
        v20 = *(_BYTE *)(v22 + 200);
        v18 = *(_QWORD *)(v22 + 184);
        if ( dword_4D0425C && !a3 )
        {
          v11 = 0;
          if ( v20 )
          {
            v19 = 0;
            v11 = 0;
            goto LABEL_46;
          }
        }
        else if ( v20 )
        {
          v17 = 0;
          v19 = 0;
          v14 = 1;
          a3 = 0;
          goto LABEL_22;
        }
        if ( v18 )
        {
          v17 = 0;
          v19 = 0;
          v20 = 0;
          v14 = 1;
          a3 = 0;
        }
        else
        {
          v31 = *(__int64 **)(v22 + 192);
          if ( v31 && (v32 = *v31) != 0 && (v33 = *(char **)(v32 + 8), !memcmp(v33, "operator \"\"", 0xBu)) )
          {
            v19 = v33 + 11;
            v17 = 0;
            a3 = 0;
            v20 = 0;
            v14 = 1;
          }
          else
          {
            a3 = v22;
            v19 = 0;
            v17 = 0;
            v20 = 0;
            v14 = 0;
          }
        }
        goto LABEL_22;
      case 13:
        v17 = *(_QWORD *)(v22 + 184);
        break;
      case 2:
        v29 = *(_BYTE **)(v22 + 8);
        if ( v29 && *v29 == 126 )
        {
          a3 = *(_QWORD *)(v11 + 8);
          v17 = *(_QWORD *)(v11 + 16);
          if ( a3 )
          {
            if ( v17 )
              goto LABEL_28;
            goto LABEL_43;
          }
          v27 = *(_BYTE *)(v11 + 33);
          v18 = 0;
          v19 = 0;
          v20 = 0;
          v14 = 0;
          goto LABEL_69;
        }
        a3 = v22;
        v18 = 0;
        v17 = 0;
        v19 = 0;
        v20 = 0;
        v14 = 0;
LABEL_22:
        if ( !v11 )
          goto LABEL_27;
        if ( *(_QWORD *)(v11 + 8) )
        {
LABEL_24:
          if ( !a3 && !v14 )
          {
            a3 = 0;
            goto LABEL_27;
          }
          goto LABEL_80;
        }
        v27 = *(_BYTE *)(v11 + 33);
LABEL_69:
        if ( (v27 & 1) == 0 )
          goto LABEL_27;
        goto LABEL_24;
      default:
        v17 = 0;
        break;
    }
    goto LABEL_87;
  }
  if ( v13 == 3 )
  {
    v10 = 1;
    goto LABEL_17;
  }
  if ( v13 != 20 )
  {
    v10 = 1;
    if ( v13 == 4 )
    {
      v15 = *(_QWORD *)(v8 + 56);
      if ( !dword_4D0425C )
      {
        a3 = *(_QWORD *)(v8 + 56);
        v10 = 1;
LABEL_96:
        if ( (*(_BYTE *)(v15 + 89) & 8) != 0 )
          v21 = *(_QWORD *)(v15 + 24);
        else
          v21 = *(_QWORD *)(v15 + 8);
        if ( v21 )
        {
          v17 = 0;
          v18 = 0;
          v19 = 0;
LABEL_18:
          v20 = 0;
          v14 = 0;
          v21 = 0;
          goto LABEL_22;
        }
        v20 = *(_BYTE *)(*(_QWORD *)(v8 + 56) + 145LL) & 1;
        if ( v20 )
        {
          v50 = v10;
          v45 = v15;
          v30 = sub_7E1620("__captured_this");
          v18 = 0;
          v19 = 0;
          v21 = 0;
          v10 = v50;
          v14 = 0;
          *(_QWORD *)(v45 + 8) = v30;
          v17 = 0;
          v20 = 0;
          goto LABEL_22;
        }
LABEL_90:
        v17 = 0;
        v18 = 0;
        v19 = 0;
        v14 = 0;
        goto LABEL_22;
      }
      v16 = a3;
      v10 = 1;
      a3 = *(_QWORD *)(v8 + 56);
      goto LABEL_38;
    }
LABEL_21:
    v17 = 0;
    v18 = 0;
    v19 = 0;
    v20 = 0;
    v14 = 0;
    v21 = 0;
    a3 = 0;
    goto LABEL_22;
  }
  v10 = 1;
LABEL_57:
  if ( !dword_4D0425C )
  {
LABEL_89:
    a3 = *(_QWORD *)(v8 + 56);
    v20 = *(_BYTE *)(a3 + 174);
    v21 = *(_QWORD *)(a3 + 240);
    if ( !v20 )
      goto LABEL_90;
    goto LABEL_61;
  }
  if ( v11 && (*(_BYTE *)(v11 + 33) & 2) != 0 )
  {
    a3 = *(_QWORD *)(v8 + 56);
    v20 = *(_BYTE *)(a3 + 174);
    v21 = *(_QWORD *)(a3 + 240);
    if ( !v20 )
      goto LABEL_31;
    goto LABEL_61;
  }
  if ( v14 )
  {
LABEL_105:
    v53 = 1;
    goto LABEL_21;
  }
  if ( a3 )
  {
    a3 = *(_QWORD *)(v8 + 56);
    v20 = *(_BYTE *)(a3 + 174);
    v21 = *(_QWORD *)(a3 + 240);
    if ( v20 )
    {
LABEL_61:
      if ( v20 != 2 )
      {
        if ( v20 == 5 )
        {
          v20 = *(_BYTE *)(a3 + 176);
          v17 = 0;
          v18 = 0;
          v19 = 0;
          v14 = 1;
          a3 = 0;
          goto LABEL_22;
        }
        if ( v20 != 4 )
        {
          if ( v20 != 3 )
            sub_721090();
          if ( dword_4D0425C )
          {
            v18 = 0;
            v14 = 0;
          }
          else
          {
            v14 = 1;
            v18 = *(_QWORD *)(*(_QWORD *)(a3 + 152) + 160LL);
          }
          v17 = 0;
          v19 = 0;
          v20 = 0;
          a3 = 0;
          v53 = 1;
          goto LABEL_22;
        }
        v20 = *(_BYTE *)(a3 + 89) & 0x40;
        if ( v20 )
        {
          v17 = 0;
          v18 = 0;
          v19 = 0;
          v20 = 0;
          v14 = 1;
          a3 = 0;
          goto LABEL_22;
        }
        if ( (*(_BYTE *)(a3 + 89) & 8) != 0 )
        {
          a3 = *(_QWORD *)(a3 + 24);
          if ( a3 )
          {
LABEL_122:
            v19 = (char *)(a3 + 11);
            v17 = 0;
            v18 = 0;
            v14 = 1;
            a3 = 0;
            goto LABEL_22;
          }
        }
        else
        {
          a3 = *(_QWORD *)(a3 + 8);
          if ( a3 )
            goto LABEL_122;
        }
        v17 = 0;
        v18 = 0;
        v19 = 0;
        v14 = 1;
        goto LABEL_22;
      }
      v17 = *(_QWORD *)(*(_QWORD *)(a3 + 40) + 32LL);
LABEL_87:
      v18 = 0;
      v19 = 0;
      v20 = 0;
      v14 = 0;
      a3 = 0;
      goto LABEL_22;
    }
  }
  else
  {
    if ( !a2 )
      goto LABEL_105;
    v51 = v10;
    v41 = a5;
    while ( 1 )
    {
      v55 = 0;
      v34 = (_QWORD *)sub_80AAC0(v7, &v55);
      sub_76C7C0((__int64)v56);
      v56[0] = sub_80A340;
      v56[2] = sub_80A840;
      v57 = 0;
      v58 = 1;
      sub_76CDC0(v34, (__int64)v56, v35, v36, v37);
      v7 = *(_QWORD *)(v7 + 16);
      if ( v57 )
        break;
      if ( !v7 )
      {
        v10 = v51;
        v7 = a2;
        a5 = v41;
        goto LABEL_105;
      }
    }
    v14 = 0;
    v10 = v51;
    v7 = a2;
    a5 = v41;
    a3 = *(_QWORD *)(v8 + 56);
    v20 = *(_BYTE *)(a3 + 174);
    v21 = *(_QWORD *)(a3 + 240);
    if ( v20 )
      goto LABEL_61;
    if ( !dword_4D0425C )
      goto LABEL_90;
  }
  if ( !v11 )
    goto LABEL_31;
  v20 = *(_BYTE *)(v11 + 33) & 2;
  if ( v20 )
    goto LABEL_31;
  v19 = *(char **)(v11 + 8);
  if ( v19 )
  {
    v17 = 0;
    v18 = 0;
    v19 = 0;
  }
  else
  {
    if ( (*(_BYTE *)(v11 + 33) & 1) == 0 )
      goto LABEL_31;
    v17 = 0;
    v18 = 0;
  }
LABEL_80:
  v38 = v10;
  v39 = v17;
  v40 = v18;
  v42 = v19;
  v49 = v20;
  v44 = v14;
  v47 = v21;
  sub_8128F0(v11, a5);
  v21 = v47;
  v14 = v44;
  v20 = v49;
  v19 = v42;
  v18 = v40;
  v17 = v39;
  v10 = v38;
LABEL_27:
  if ( v17 )
  {
LABEL_28:
    sub_8129A0(v17, v11, a5);
    return;
  }
  if ( !v14 )
  {
    if ( a3 )
    {
LABEL_31:
      sub_812380(a3, v21, v11, a5);
      return;
    }
LABEL_43:
    sub_816460(v8, a4, v53, a5);
    return;
  }
LABEL_46:
  if ( !v7 )
  {
    v26 = 0;
    goto LABEL_53;
  }
  v24 = v7;
  v25 = 0;
  do
  {
    if ( (*(_BYTE *)(v24 + 25) & 0x10) != 0 )
      break;
    v24 = *(_QWORD *)(v24 + 16);
    ++v25;
  }
  while ( v24 );
  if ( (unsigned __int8)v20 > 7u )
  {
    if ( v20 != 11 )
      goto LABEL_52;
  }
  else if ( (unsigned __int8)v20 <= 4u )
  {
    goto LABEL_52;
  }
  if ( v25 == 1 )
  {
    v26 = ((*(_BYTE *)(v7 + 26) & 4) != 0) + 1;
    goto LABEL_53;
  }
LABEL_52:
  v26 = v25;
LABEL_53:
  sub_812220(v20, v26, v18, v19, v21, v11, v10, a5);
}
