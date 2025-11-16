// Function: sub_CBD060
// Address: 0xcbd060
//
char *__fastcall sub_CBD060(__int64 *a1, char *a2, char *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v6; // r14
  __int64 *v8; // rbx
  __int64 v9; // rdi
  char *v10; // r8
  __int64 v11; // rdx
  _BYTE *v12; // rax
  int v13; // r15d
  __int64 v14; // r12
  __int64 v15; // rax
  _BYTE *v16; // r13
  __int64 v17; // r12
  __int64 v18; // r14
  __int64 v19; // r11
  int v20; // r10d
  int v21; // r8d
  bool v22; // cc
  int v23; // eax
  int v24; // edi
  int v25; // eax
  char *v26; // rsi
  int v27; // eax
  int v28; // eax
  bool v29; // cc
  __int64 v30; // rcx
  __int64 *v31; // r15
  int v32; // r13d
  __int64 v33; // rdi
  int v34; // ebx
  __int64 v35; // rax
  int v36; // eax
  __int64 v37; // rax
  int v38; // eax
  int v39; // eax
  int v41; // [rsp+10h] [rbp-70h]
  __int64 v42; // [rsp+10h] [rbp-70h]
  __int64 v43; // [rsp+10h] [rbp-70h]
  __int64 v44; // [rsp+10h] [rbp-70h]
  __int64 v45; // [rsp+10h] [rbp-70h]
  __int64 v46; // [rsp+18h] [rbp-68h]
  int v47; // [rsp+18h] [rbp-68h]
  int v48; // [rsp+18h] [rbp-68h]
  int v49; // [rsp+18h] [rbp-68h]
  int v50; // [rsp+18h] [rbp-68h]
  int v51; // [rsp+18h] [rbp-68h]
  void *dest; // [rsp+28h] [rbp-58h]
  void *s2; // [rsp+30h] [rbp-50h]
  char *v55; // [rsp+38h] [rbp-48h]
  int n; // [rsp+40h] [rbp-40h]
  int na; // [rsp+40h] [rbp-40h]
  size_t nb; // [rsp+40h] [rbp-40h]
  char *v59; // [rsp+48h] [rbp-38h]

  v5 = a4;
  v6 = a4;
  v8 = a1;
  v59 = a2;
  v9 = *a1;
  if ( a4 >= a5 )
    goto LABEL_6;
  v10 = a2;
  while ( 1 )
  {
    v11 = *(_QWORD *)(*(_QWORD *)(v9 + 8) + 8 * v6);
    v5 = v6;
    if ( ((((unsigned int)v11 & 0xF8000000) - 1744830464LL) & 0xFFFFFFFFF0000000LL) != 0 )
      break;
LABEL_62:
    if ( a5 == ++v6 )
    {
      v59 = v10;
      v5 = a5;
      goto LABEL_6;
    }
  }
  if ( (v11 & 0xF8000000) != 0x10000000 )
  {
    v59 = v10;
LABEL_6:
    n = 128;
    s2 = (void *)v8[13];
    dest = (void *)v8[12];
    if ( (char *)v8[4] != v59 )
      n = *(v59 - 1);
    v12 = memset((void *)v8[10], 0, *(_QWORD *)(v9 + 48));
    v12[v5] = 1;
    v55 = 0;
    v13 = n;
    v14 = sub_CBB420(*v8, v6, a5, (__int64)v12, 132, (__int64)v12);
    v15 = a5;
    v16 = (_BYTE *)v14;
    v17 = v6;
    v18 = v15;
    while ( (char *)v8[5] != v59 )
    {
      v19 = *v8;
      na = *v59;
      if ( v13 == 10 )
      {
        v21 = *(_DWORD *)(v19 + 40) & 8;
        if ( !v21 )
          goto LABEL_36;
LABEL_50:
        v20 = *(_DWORD *)(v19 + 76);
        v21 = 129;
        goto LABEL_11;
      }
      v20 = 0;
      v21 = 0;
      if ( v13 == 128 && (v8[1] & 1) == 0 )
        goto LABEL_50;
LABEL_11:
      if ( na != 10 )
      {
        if ( na != 128 )
          goto LABEL_13;
        v28 = *((_DWORD *)v8 + 2);
LABEL_33:
        na = 128;
        if ( (v28 & 2) != 0 )
        {
LABEL_13:
          v22 = v20 <= 0;
          if ( v20 )
          {
LABEL_42:
            if ( !v22 )
            {
LABEL_43:
              v49 = v13;
              v30 = (__int64)v16;
              v31 = v8;
              v32 = v21;
              v33 = v19;
              v34 = v20;
              while ( 1 )
              {
                v35 = sub_CBB420(v33, v17, v18, v30, v32, v30);
                v30 = v35;
                if ( !--v34 )
                  break;
                v33 = *v31;
              }
              v8 = v31;
              v21 = v32;
              v13 = v49;
              v16 = (_BYTE *)v35;
              v19 = *v8;
            }
          }
LABEL_14:
          if ( v21 == 129 )
          {
            if ( na != 128 )
              goto LABEL_16;
            goto LABEL_18;
          }
          goto LABEL_35;
        }
        goto LABEL_34;
      }
      if ( (*(_BYTE *)(v19 + 40) & 8) == 0 )
      {
        v22 = v20 <= 0;
        if ( v20 )
          goto LABEL_42;
        goto LABEL_14;
      }
LABEL_34:
      v20 += *(_DWORD *)(v19 + 80);
      v21 = (v21 == 129) + 130;
      v29 = v20 <= 0;
      if ( !v20 )
        goto LABEL_35;
LABEL_74:
      if ( !v29 )
        goto LABEL_43;
LABEL_35:
      if ( v13 == 128 )
        goto LABEL_23;
LABEL_36:
      v43 = v19;
      v48 = v21;
      v25 = isalnum((unsigned __int8)v13);
      v21 = v48;
      v19 = v43;
      if ( v13 == 95 )
        goto LABEL_47;
      if ( !v25 )
      {
        v24 = (unsigned __int8)v13;
        if ( na == 128 )
        {
LABEL_20:
          v42 = v19;
          v47 = v21;
          v25 = isalnum(v24);
          v21 = v47;
          v19 = v42;
          goto LABEL_21;
        }
LABEL_16:
        v46 = v19;
        v41 = v21;
        v23 = isalnum((unsigned __int8)na);
        v19 = v46;
        if ( na == 95 || (v21 = v41, v23) )
        {
          if ( v13 == 128 )
            goto LABEL_68;
          v44 = v46;
          v50 = v23;
          v38 = isalnum((unsigned __int8)v13);
          v19 = v44;
          if ( !v38 && v13 != 95 )
            goto LABEL_68;
          if ( na == 95 || v50 )
            goto LABEL_68;
          goto LABEL_56;
        }
LABEL_18:
        if ( v13 == 128 )
          goto LABEL_23;
        v24 = (unsigned __int8)v13;
        goto LABEL_20;
      }
LABEL_21:
      if ( v13 != 95 && !v25 )
      {
LABEL_23:
        v13 = na;
        goto LABEL_24;
      }
LABEL_47:
      v36 = 134;
      if ( v21 == 130 )
        goto LABEL_48;
      v13 = 128;
      if ( na != 128 )
      {
        v45 = v19;
        v51 = v21;
        v39 = isalnum((unsigned __int8)na);
        v19 = v45;
        if ( na == 95 || v39 )
        {
          if ( (unsigned int)(v51 - 133) > 1 )
            goto LABEL_23;
LABEL_68:
          v36 = 133;
LABEL_48:
          v37 = sub_CBB420(v19, v17, v18, (__int64)v16, v36, (__int64)v16);
          v19 = *v8;
          v13 = na;
          v16 = (_BYTE *)v37;
          goto LABEL_24;
        }
LABEL_56:
        v36 = 134;
        goto LABEL_48;
      }
LABEL_24:
      v26 = v55;
      if ( v16[v18] )
        v26 = v59;
      v55 = v26;
      nb = *(_QWORD *)(v19 + 48);
      v27 = memcmp(v16, s2, nb);
      if ( v59 == a3 || !v27 )
        return v55;
      memmove(dest, v16, nb);
      memmove(v16, s2, *(_QWORD *)(*v8 + 48));
      ++v59;
      v16 = (_BYTE *)sub_CBB420(*v8, v17, v18, (__int64)dest, v13, (__int64)v16);
    }
    v19 = *v8;
    if ( v13 == 10 )
    {
      v21 = *(_DWORD *)(v19 + 40) & 8;
      if ( !v21 )
      {
        na = 128;
        if ( (v8[1] & 2) != 0 )
          goto LABEL_36;
        v20 = *(_DWORD *)(v19 + 80);
        v21 = 130;
        v29 = v20 <= 0;
        if ( !v20 )
          goto LABEL_35;
        goto LABEL_74;
      }
      v20 = *(_DWORD *)(v19 + 76);
      v28 = *((_DWORD *)v8 + 2);
      v21 = 129;
      goto LABEL_33;
    }
    if ( v13 != 128 )
    {
      v28 = *((_DWORD *)v8 + 2);
      v20 = 0;
      v21 = 0;
      goto LABEL_33;
    }
    v28 = *((_DWORD *)v8 + 2);
    if ( (v28 & 1) != 0 )
    {
      v20 = 0;
      v21 = 0;
      goto LABEL_33;
    }
    na = 128;
    goto LABEL_50;
  }
  if ( a3 != v10 && *v10 == (_BYTE)v11 )
  {
    ++v10;
    goto LABEL_62;
  }
  return 0;
}
