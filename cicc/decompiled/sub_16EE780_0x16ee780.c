// Function: sub_16EE780
// Address: 0x16ee780
//
char *__fastcall sub_16EE780(__int64 *a1, char *a2, char *a3, __int64 a4, __int64 a5)
{
  int v6; // r12d
  __int64 *v7; // rbx
  _BYTE *v8; // rax
  _BYTE *i; // r14
  __int64 v10; // r11
  int v11; // r10d
  int v12; // r15d
  bool v13; // cc
  int v14; // eax
  int v15; // edi
  int v16; // eax
  char *v17; // rsi
  size_t v18; // r15
  int v19; // eax
  bool v20; // cc
  __int64 v21; // rcx
  int v22; // r14d
  __int64 v23; // rdi
  __int64 *v24; // r15
  int v25; // ebx
  __int64 v26; // rax
  int v27; // r8d
  __int64 v28; // rax
  int v29; // eax
  int v30; // eax
  __int64 v32; // [rsp+8h] [rbp-78h]
  __int64 v33; // [rsp+10h] [rbp-70h]
  __int64 v34; // [rsp+10h] [rbp-70h]
  __int64 v35; // [rsp+10h] [rbp-70h]
  int v36; // [rsp+10h] [rbp-70h]
  __int64 v37; // [rsp+10h] [rbp-70h]
  void *dest; // [rsp+18h] [rbp-68h]
  void *s2; // [rsp+30h] [rbp-50h]
  char *v42; // [rsp+38h] [rbp-48h]
  int v43; // [rsp+44h] [rbp-3Ch]
  char *v44; // [rsp+48h] [rbp-38h]

  v6 = 128;
  v7 = a1;
  v44 = a2;
  s2 = (void *)a1[13];
  dest = (void *)a1[12];
  if ( (char *)a1[4] != a2 )
    v6 = *(a2 - 1);
  v8 = memset((void *)a1[10], 0, *(_QWORD *)(*a1 + 48));
  v8[a4] = 1;
  v42 = 0;
  for ( i = (_BYTE *)sub_16ECBD0(*a1, a4, a5, (__int64)v8, 132, (__int64)v8);
        ;
        i = (_BYTE *)sub_16ECBD0(*v7, a4, a5, (__int64)dest, v6, (__int64)i) )
  {
    if ( (char *)v7[5] == v44 )
    {
      v10 = *v7;
      if ( v6 == 10 )
      {
        v12 = *(_DWORD *)(v10 + 40) & 8;
        if ( !v12 )
        {
          v43 = 128;
          if ( (v7[1] & 2) != 0 )
            goto LABEL_31;
          v11 = *(_DWORD *)(v10 + 80);
          v12 = 130;
          v20 = v11 <= 0;
          if ( !v11 )
            goto LABEL_30;
          goto LABEL_63;
        }
        v11 = *(_DWORD *)(v10 + 76);
        v19 = *((_DWORD *)v7 + 2);
        v12 = 129;
LABEL_28:
        v43 = 128;
        if ( (v19 & 2) == 0 )
          goto LABEL_29;
LABEL_8:
        v13 = v11 <= 0;
        if ( !v11 )
        {
LABEL_9:
          if ( v12 == 129 )
          {
            if ( v43 != 128 )
              goto LABEL_11;
            goto LABEL_13;
          }
          goto LABEL_30;
        }
LABEL_37:
        if ( !v13 )
        {
LABEL_38:
          v21 = (__int64)i;
          v22 = v12;
          v23 = v10;
          v24 = v7;
          v25 = v11;
          while ( 1 )
          {
            v26 = sub_16ECBD0(v23, a4, a5, v21, v22, v21);
            v21 = v26;
            if ( !--v25 )
              break;
            v23 = *v24;
          }
          v7 = v24;
          v12 = v22;
          i = (_BYTE *)v26;
          v10 = *v7;
        }
        goto LABEL_9;
      }
      if ( v6 != 128 )
      {
        v19 = *((_DWORD *)v7 + 2);
        v11 = 0;
        v12 = 0;
        goto LABEL_28;
      }
      v19 = *((_DWORD *)v7 + 2);
      if ( (v19 & 1) != 0 )
      {
        v11 = 0;
        v12 = 0;
        goto LABEL_28;
      }
      v43 = 128;
      goto LABEL_45;
    }
    v10 = *v7;
    v43 = *v44;
    if ( v6 == 10 )
    {
      v12 = *(_DWORD *)(v10 + 40) & 8;
      if ( !v12 )
        goto LABEL_31;
LABEL_45:
      v11 = *(_DWORD *)(v10 + 76);
      v12 = 129;
      goto LABEL_6;
    }
    v11 = 0;
    v12 = 0;
    if ( v6 == 128 && (v7[1] & 1) == 0 )
      goto LABEL_45;
LABEL_6:
    if ( v43 != 10 )
    {
      if ( v43 != 128 )
        goto LABEL_8;
      v19 = *((_DWORD *)v7 + 2);
      goto LABEL_28;
    }
    if ( (*(_BYTE *)(v10 + 40) & 8) == 0 )
    {
      v13 = v11 <= 0;
      if ( !v11 )
        goto LABEL_9;
      goto LABEL_37;
    }
LABEL_29:
    v11 += *(_DWORD *)(v10 + 80);
    v12 = (v12 == 129) + 130;
    v20 = v11 <= 0;
    if ( !v11 )
      goto LABEL_30;
LABEL_63:
    if ( !v20 )
      goto LABEL_38;
LABEL_30:
    if ( v6 == 128 )
      goto LABEL_18;
LABEL_31:
    v35 = v10;
    v16 = isalnum((unsigned __int8)v6);
    v10 = v35;
    if ( v6 == 95 )
      goto LABEL_42;
    if ( !v16 )
    {
      v15 = (unsigned __int8)v6;
      if ( v43 == 128 )
      {
LABEL_15:
        v34 = v10;
        v16 = isalnum(v15);
        v10 = v34;
        goto LABEL_16;
      }
LABEL_11:
      v33 = v10;
      v14 = isalnum((unsigned __int8)v43);
      v10 = v33;
      if ( v43 == 95 || v14 )
      {
        if ( v6 == 128 )
          goto LABEL_58;
        v32 = v33;
        v36 = v14;
        v29 = isalnum((unsigned __int8)v6);
        v10 = v32;
        if ( v6 != 95 && !v29 )
          goto LABEL_58;
        if ( v43 == 95 || v36 )
          goto LABEL_58;
        goto LABEL_51;
      }
LABEL_13:
      if ( v6 == 128 )
        goto LABEL_18;
      v15 = (unsigned __int8)v6;
      goto LABEL_15;
    }
LABEL_16:
    if ( v6 != 95 && !v16 )
    {
LABEL_18:
      v6 = v43;
      goto LABEL_19;
    }
LABEL_42:
    v27 = 134;
    if ( v12 == 130 )
      goto LABEL_43;
    v6 = 128;
    if ( v43 != 128 )
    {
      v37 = v10;
      v30 = isalnum((unsigned __int8)v43);
      v10 = v37;
      if ( v43 == 95 || v30 )
      {
        if ( (unsigned int)(v12 - 133) > 1 )
          goto LABEL_18;
LABEL_58:
        v27 = 133;
LABEL_43:
        v28 = sub_16ECBD0(v10, a4, a5, (__int64)i, v27, (__int64)i);
        v10 = *v7;
        v6 = v43;
        i = (_BYTE *)v28;
        goto LABEL_19;
      }
LABEL_51:
      v27 = 134;
      goto LABEL_43;
    }
LABEL_19:
    v17 = v42;
    if ( i[a5] )
      v17 = v44;
    v18 = *(_QWORD *)(v10 + 48);
    v42 = v17;
    if ( !memcmp(i, s2, v18) || v44 == a3 )
      break;
    memmove(dest, i, v18);
    memmove(i, s2, *(_QWORD *)(*v7 + 48));
    ++v44;
  }
  return v17;
}
