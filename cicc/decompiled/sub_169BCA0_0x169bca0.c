// Function: sub_169BCA0
// Address: 0x169bca0
//
__int64 __fastcall sub_169BCA0(__int16 **a1, char *a2, __int64 a3, int a4)
{
  char *v5; // rbx
  char *v8; // r13
  char *v9; // rcx
  int v10; // r9d
  char *v11; // r8
  int v12; // eax
  char *i; // rdx
  char *v14; // rsi
  int v15; // eax
  char v16; // r10
  char *v17; // rax
  unsigned int v18; // esi
  unsigned int v19; // r12d
  int v20; // eax
  __int64 result; // rax
  char *v22; // r10
  __int64 v23; // rax
  unsigned int v24; // r11d
  unsigned int v25; // r8d
  __int64 v26; // rsi
  __int16 **v27; // r10
  unsigned int v28; // r14d
  int v29; // r12d
  int v30; // edi
  int v31; // edx
  int v32; // ecx
  int v33; // eax
  int v34; // r9d
  int v35; // r12d
  __int16 **v36; // r14
  unsigned int v37; // edx
  unsigned int v38; // [rsp+8h] [rbp-58h]
  unsigned int v39; // [rsp+8h] [rbp-58h]
  __int16 **v40; // [rsp+8h] [rbp-58h]
  __int16 **v41; // [rsp+10h] [rbp-50h]
  unsigned int v42; // [rsp+10h] [rbp-50h]
  unsigned int v43; // [rsp+10h] [rbp-50h]
  unsigned int v44; // [rsp+18h] [rbp-48h]
  unsigned int v45; // [rsp+1Ch] [rbp-44h]
  unsigned int v46; // [rsp+1Ch] [rbp-44h]
  unsigned int v47; // [rsp+1Ch] [rbp-44h]
  unsigned int v48; // [rsp+1Ch] [rbp-44h]
  unsigned __int64 v49; // [rsp+20h] [rbp-40h]
  __int64 v50; // [rsp+28h] [rbp-38h]
  unsigned int v51; // [rsp+28h] [rbp-38h]
  unsigned int v52; // [rsp+28h] [rbp-38h]

  v5 = &a2[a3];
  if ( &a2[a3] == a2 )
    goto LABEL_38;
  v8 = a2;
  v9 = a2;
  while ( 1 )
  {
    v10 = *v9;
    if ( *v9 != 48 )
      break;
    if ( v5 == ++v9 )
      goto LABEL_66;
  }
  if ( v5 == v9 )
    goto LABEL_66;
  if ( (_BYTE)v10 == 46 )
  {
    v11 = v9 + 1;
    if ( v5 != v9 + 1 )
    {
      while ( 1 )
      {
        v10 = *v11;
        if ( *v11 != 48 )
          break;
        if ( v5 == ++v11 )
          goto LABEL_65;
      }
      if ( v5 == v11 )
        goto LABEL_38;
      goto LABEL_8;
    }
LABEL_65:
    v9 = &a2[a3];
LABEL_66:
    v11 = v9;
    i = v9;
    v20 = 0;
    v19 = 0;
    goto LABEL_33;
  }
  v11 = v9;
  v9 = &a2[a3];
LABEL_8:
  v12 = (char)v10;
  for ( i = v11; ; ++i )
  {
    if ( (_BYTE)v12 != 46 )
    {
      if ( (unsigned int)(v12 - 48) > 9 )
        break;
      goto LABEL_10;
    }
    v14 = i + 1;
    if ( v5 == i + 1 )
      goto LABEL_64;
    v15 = i[1];
    v9 = i++;
    if ( (unsigned int)(v15 - 48) > 9 )
      break;
LABEL_10:
    if ( v5 == i + 1 )
    {
      i = v5;
      v19 = 0;
      goto LABEL_26;
    }
    v12 = i[1];
  }
  if ( v5 == i )
  {
    if ( v11 == v5 )
      goto LABEL_38;
    v14 = v5;
    i = v9;
LABEL_64:
    v9 = i;
    v19 = 0;
    i = v14;
    do
LABEL_28:
      --i;
    while ( a2 != i && (*i == 48 || a2 != i && *i == 46) );
LABEL_32:
    v19 += (__int16)((_WORD)v9 - ((i < v9) + (_WORD)i));
    v20 = v19 + (__int16)((_WORD)i - (_WORD)v11 - (i > v9 && v11 < v9));
    goto LABEL_33;
  }
  v16 = i[1];
  if ( ((v16 - 43) & 0xFD) != 0 )
  {
    v17 = i + 2;
    v18 = v16 - 48;
    v19 = v18;
    if ( v5 != i + 2 )
      goto LABEL_18;
  }
  else
  {
    v17 = i + 3;
    v18 = i[2] - 48;
    v19 = v18;
    if ( v5 != i + 3 )
    {
LABEL_18:
      v19 = *v17 + 10 * v18 - 48;
      if ( v18 <= 0x5DBF )
      {
        while ( v5 != ++v17 )
        {
          if ( v19 > 0x5DBF )
            goto LABEL_49;
          v19 = *v17 + 10 * v19 - 48;
        }
      }
      else
      {
LABEL_49:
        v19 = 24000;
      }
    }
    if ( v16 == 45 )
      v19 = -v19;
  }
  if ( v5 == v9 )
  {
    if ( v11 != i )
    {
      v9 = i;
LABEL_27:
      if ( a2 == i )
        goto LABEL_32;
      goto LABEL_28;
    }
    v20 = 0;
LABEL_33:
    if ( v5 == v11 || (unsigned int)(*v11 - 48) > 9 )
      goto LABEL_38;
    if ( v20 > 51084 )
      return sub_1698D70((__int64)a1, a4);
    if ( v20 >= -51082 )
    {
      v22 = i;
      LODWORD(i) = (_DWORD)v11;
      goto LABEL_51;
    }
LABEL_61:
    v51 = a4;
    *((_BYTE *)a1 + 18) = *((_BYTE *)a1 + 18) & 0xF8 | 2;
    sub_1698870((__int64)a1);
    return sub_1698EC0(a1, v51, 1u);
  }
LABEL_26:
  if ( v11 != i )
    goto LABEL_27;
  if ( (unsigned int)(v10 - 48) > 9 )
  {
LABEL_38:
    *((_BYTE *)a1 + 18) = *((_BYTE *)a1 + 18) & 0xF8 | 3;
    return 0;
  }
  v22 = i;
  v20 = 0;
LABEL_51:
  if ( 28738 * (v20 + 1) <= 8651 * ((*a1)[1] - *((_DWORD *)*a1 + 1)) )
    goto LABEL_61;
  if ( 42039 * (v20 - 1) >= 12655 * **a1 )
    return sub_1698D70((__int64)a1, a4);
  v45 = a4;
  v49 = (unsigned __int64)v22;
  v23 = sub_2207820(8LL * (((196 * ((int)v22 - (int)i + 1) / 0x3Bu + 64) >> 6) + 1));
  v24 = v45;
  v25 = 0;
  v50 = 0;
  v26 = v23;
  v27 = a1;
  v28 = v19;
  v29 = 1;
LABEL_54:
  while ( 2 )
  {
    v30 = 19;
    v31 = 1;
    v32 = 0;
    while ( 1 )
    {
      v33 = *v8;
      if ( *v8 != 46 )
        goto LABEL_55;
      if ( v5 == v8 + 1 )
        break;
      v33 = *++v8;
LABEL_55:
      ++v8;
      v31 *= 10;
      v32 = v33 - 48 + 10 * v32;
      if ( (unsigned __int64)v8 > v49 )
      {
        v34 = v29;
        v39 = v24;
        v35 = v28;
        v36 = v27;
        v42 = v34;
        v47 = v25;
        sub_16A7890(v26, v26, v31, v32, v25, v34, 0);
        v37 = v47;
        v24 = v39;
        if ( *(_QWORD *)(v26 + 8 * v50) )
          v37 = v42;
        goto LABEL_71;
      }
      if ( !--v30 )
      {
        v38 = v24;
        v41 = v27;
        v46 = v25;
        v44 = v29;
        sub_16A7890(v26, v26, v31, v32, v25, v29, 0);
        v25 = v46;
        v27 = v41;
        v24 = v38;
        if ( *(_QWORD *)(v26 + 8 * v50) )
        {
          v25 = v29++;
          v50 = v44;
        }
        goto LABEL_54;
      }
    }
    v40 = v27;
    v43 = v24;
    v48 = v25;
    sub_16A7890(v26, v26, v31, v32, v25, v29, 0);
    v25 = v48;
    v24 = v43;
    v27 = v40;
    if ( *(_QWORD *)(v26 + 8 * v50) )
      v25 = v29;
    if ( v49 >= (unsigned __int64)v5 )
    {
      v8 = v5;
      v29 = v25 + 1;
      v50 = v25;
      continue;
    }
    break;
  }
  v35 = v28;
  v37 = v25;
  v36 = v40;
LABEL_71:
  *((_BYTE *)v36 + 18) = *((_BYTE *)v36 + 18) & 0xF8 | 2;
  result = sub_169B680(v36, v26, v37, v35, v24);
  if ( v26 )
  {
    v52 = result;
    j_j___libc_free_0_0(v26);
    return v52;
  }
  return result;
}
