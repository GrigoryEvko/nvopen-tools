// Function: sub_123B840
// Address: 0x123b840
//
__int64 __fastcall sub_123B840(__int64 a1, _QWORD *a2)
{
  unsigned int v5; // eax
  int v6; // eax
  _QWORD *v7; // rdi
  char *v8; // r9
  void *v9; // r8
  _QWORD *v10; // r13
  signed __int64 v11; // r10
  char *v12; // rsi
  char *v13; // rax
  char *v14; // rdx
  _QWORD *v15; // rdx
  signed __int64 v16; // rax
  char *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  char *v20; // rsi
  _QWORD *v21; // rcx
  char **v22; // r11
  char *v23; // rax
  unsigned __int64 v24; // r9
  __int64 v25; // rax
  char *v26; // r8
  size_t v27; // rdx
  _QWORD *v28; // rax
  _QWORD *v29; // rdx
  char *v30; // r10
  char *v31; // r8
  signed __int64 v32; // r9
  _QWORD *v33; // rcx
  _BOOL8 v34; // rdi
  __int64 v35; // rax
  int v36; // eax
  unsigned __int64 v37; // rsi
  unsigned __int64 v38; // rsi
  _QWORD *v39; // rax
  signed __int64 v40; // rdi
  char **v41; // [rsp+0h] [rbp-E0h]
  _QWORD *v42; // [rsp+8h] [rbp-D8h]
  char *v43; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v44; // [rsp+10h] [rbp-D0h]
  _QWORD *v45; // [rsp+10h] [rbp-D0h]
  char **v46; // [rsp+18h] [rbp-C8h]
  char *v47; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v48; // [rsp+18h] [rbp-C8h]
  char *v49; // [rsp+20h] [rbp-C0h]
  char *v50; // [rsp+20h] [rbp-C0h]
  _QWORD *v51; // [rsp+28h] [rbp-B8h]
  _QWORD *v52; // [rsp+28h] [rbp-B8h]
  signed __int64 v53; // [rsp+30h] [rbp-B0h]
  _QWORD *v54; // [rsp+30h] [rbp-B0h]
  void *src; // [rsp+40h] [rbp-A0h] BYREF
  char *v56; // [rsp+48h] [rbp-98h]
  __int64 v57; // [rsp+50h] [rbp-90h]
  __m128i v58; // [rsp+60h] [rbp-80h] BYREF
  _QWORD v59[2]; // [rsp+70h] [rbp-70h] BYREF
  _QWORD v60[4]; // [rsp+80h] [rbp-60h] BYREF
  char v61; // [rsp+A0h] [rbp-40h]
  char v62; // [rsp+A1h] [rbp-3Fh]

  if ( (unsigned __int8)sub_120AFE0(a1, 483, "expected 'resByArg' here")
    || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
  {
    return 1;
  }
  while ( 1 )
  {
    src = 0;
    v56 = 0;
    v57 = 0;
    if ( (unsigned __int8)sub_1212CC0(a1, (__int64)&src)
      || (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here")
      || (unsigned __int8)sub_120AFE0(a1, 484, "expected 'byArg here")
      || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
      || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here")
      || (unsigned __int8)sub_120AFE0(a1, 465, "expected 'kind' here")
      || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here") )
    {
      goto LABEL_60;
    }
    v5 = *(_DWORD *)(a1 + 240);
    v58.m128i_i32[0] = 0;
    v58.m128i_i64[1] = 0;
    v59[0] = 0;
    if ( v5 == 486 )
    {
      v58.m128i_i32[0] = 2;
    }
    else if ( v5 > 0x1E6 )
    {
      if ( v5 != 487 )
      {
LABEL_72:
        v38 = *(_QWORD *)(a1 + 232);
        v62 = 1;
        v60[0] = "unexpected WholeProgramDevirtResolution::ByArg kind";
        v61 = 3;
        sub_11FD800(a1 + 176, v38, (__int64)v60, 1);
        goto LABEL_60;
      }
      v58.m128i_i32[0] = 3;
    }
    else if ( v5 != 479 )
    {
      if ( v5 != 485 )
        goto LABEL_72;
      v58.m128i_i32[0] = 1;
    }
    v6 = sub_1205200(a1 + 176);
    *(_DWORD *)(a1 + 240) = v6;
    while ( *(_DWORD *)(a1 + 240) == 4 )
    {
      v36 = sub_1205200(a1 + 176);
      *(_DWORD *)(a1 + 240) = v36;
      switch ( v36 )
      {
        case 489:
          *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
          if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here") || (unsigned __int8)sub_120BD00(a1, v59) )
            goto LABEL_60;
          break;
        case 490:
          *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
          if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
            || (unsigned __int8)sub_120BD00(a1, (_DWORD *)v59 + 1) )
          {
            goto LABEL_60;
          }
          break;
        case 488:
          *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
          if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
            || (unsigned __int8)sub_120C050(a1, &v58.m128i_i64[1]) )
          {
            goto LABEL_60;
          }
          break;
        default:
          v62 = 1;
          v37 = *(_QWORD *)(a1 + 232);
          v61 = 3;
          v60[0] = "expected optional whole program devirt field";
          sub_11FD800(a1 + 176, v37, (__int64)v60, 1);
          goto LABEL_60;
      }
    }
    if ( (unsigned __int8)sub_120AFE0(a1, 13, "expected ')' here") )
    {
LABEL_60:
      if ( src )
        j_j___libc_free_0(src, v57 - (_QWORD)src);
      return 1;
    }
    v7 = (_QWORD *)a2[2];
    if ( !v7 )
    {
      v10 = a2 + 1;
      goto LABEL_38;
    }
    v8 = v56;
    v9 = src;
    v10 = a2 + 1;
    v11 = v56 - (_BYTE *)src;
    do
    {
      v12 = (char *)v7[5];
      v13 = (char *)v7[4];
      if ( v12 - v13 > v11 )
        v12 = &v13[v11];
      v14 = (char *)src;
      if ( v13 != v12 )
      {
        while ( *(_QWORD *)v13 >= *(_QWORD *)v14 )
        {
          if ( *(_QWORD *)v13 > *(_QWORD *)v14 )
            goto LABEL_63;
          v13 += 8;
          v14 += 8;
          if ( v12 == v13 )
            goto LABEL_62;
        }
LABEL_28:
        v7 = (_QWORD *)v7[3];
        continue;
      }
LABEL_62:
      if ( v56 != v14 )
        goto LABEL_28;
LABEL_63:
      v10 = v7;
      v7 = (_QWORD *)v7[2];
    }
    while ( v7 );
    if ( v10 == a2 + 1 )
      goto LABEL_38;
    v15 = (_QWORD *)v10[4];
    v16 = v10[5] - (_QWORD)v15;
    if ( v11 > v16 )
      v8 = (char *)src + v16;
    if ( src == v8 )
    {
LABEL_74:
      if ( (_QWORD *)v10[5] != v15 )
        goto LABEL_38;
      goto LABEL_48;
    }
    v17 = (char *)src;
    while ( *(_QWORD *)v17 >= *v15 )
    {
      if ( *(_QWORD *)v17 > *v15 )
        goto LABEL_48;
      v17 += 8;
      ++v15;
      if ( v8 == v17 )
        goto LABEL_74;
    }
LABEL_38:
    v51 = v10;
    v18 = sub_22077B0(80);
    v20 = (char *)src;
    v21 = a2 + 1;
    v10 = (_QWORD *)v18;
    v22 = (char **)(v18 + 32);
    v23 = v56;
    v10[4] = 0;
    v10[5] = 0;
    v24 = v23 - v20;
    v10[6] = 0;
    if ( v23 == v20 )
    {
      v53 = 0;
      v27 = 0;
      v26 = 0;
    }
    else
    {
      v53 = v23 - v20;
      if ( v24 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4261EA(80, v20, v19);
      v46 = v22;
      v25 = sub_22077B0(v24);
      v20 = (char *)src;
      v21 = a2 + 1;
      v26 = (char *)v25;
      v23 = v56;
      v22 = v46;
      v24 = v56 - (_BYTE *)src;
      v27 = v56 - (_BYTE *)src;
    }
    v10[4] = v26;
    v10[5] = v26;
    v10[6] = &v26[v53];
    if ( v20 == v23 )
    {
      v10[5] = &v26[v27];
      *((_DWORD *)v10 + 14) = 0;
      v10[8] = 0;
      v10[9] = 0;
      v43 = v26;
      v45 = v21;
      v48 = v24;
      v50 = &v26[v27];
      v28 = sub_9D7C50(a2, v51, v22);
      v30 = v50;
      v32 = v48;
      v33 = v45;
      v31 = v43;
      if ( v29 )
        goto LABEL_43;
      if ( v43 )
      {
LABEL_80:
        v52 = v28;
        j_j___libc_free_0(v31, v53);
        v28 = v52;
      }
      v54 = v28;
      j_j___libc_free_0(v10, 80);
      v10 = v54;
      goto LABEL_47;
    }
    v42 = v21;
    v44 = v24;
    v47 = v26;
    v41 = v22;
    v49 = &v26[v27];
    memmove(v26, v20, v27);
    *((_DWORD *)v10 + 14) = 0;
    v10[5] = v49;
    v10[8] = 0;
    v10[9] = 0;
    v28 = sub_9D7C50(a2, v51, v41);
    v30 = v49;
    v31 = v47;
    v32 = v44;
    v33 = v42;
    if ( !v29 )
      goto LABEL_80;
LABEL_43:
    if ( v33 == v29 || v28 )
      goto LABEL_45;
    v39 = (_QWORD *)v29[4];
    v40 = v29[5] - (_QWORD)v39;
    if ( v32 > v40 )
      v30 = &v31[v40];
    if ( v31 == v30 )
    {
LABEL_89:
      v34 = v39 != (_QWORD *)v29[5];
    }
    else
    {
      while ( *(_QWORD *)v31 >= *v39 )
      {
        if ( *(_QWORD *)v31 > *v39 )
        {
          v34 = 0;
          goto LABEL_46;
        }
        v31 += 8;
        ++v39;
        if ( v30 == v31 )
          goto LABEL_89;
      }
LABEL_45:
      v34 = 1;
    }
LABEL_46:
    sub_220F040(v34, v10, v29, v33);
    ++a2[5];
LABEL_47:
    v9 = src;
LABEL_48:
    v35 = v59[0];
    *(__m128i *)(v10 + 7) = _mm_loadu_si128(&v58);
    v10[9] = v35;
    if ( v9 )
      j_j___libc_free_0(v9, v57 - (_QWORD)v9);
    if ( *(_DWORD *)(a1 + 240) != 4 )
      return sub_120AFE0(a1, 13, "expected ')' here");
    *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  }
}
