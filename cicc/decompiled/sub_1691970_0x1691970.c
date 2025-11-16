// Function: sub_1691970
// Address: 0x1691970
//
__int64 __fastcall sub_1691970(__int64 a1, const char **a2, unsigned int *a3, int a4, int a5)
{
  const char *v5; // rax
  const char *v6; // r12
  size_t v7; // r13
  int v8; // r14d
  int v9; // edx
  int v10; // ebx
  int v11; // r15d
  int v12; // ecx
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 *v15; // rbx
  __int64 *v16; // r14
  size_t v17; // rdx
  unsigned __int64 v18; // rax
  __int64 v19; // r13
  char *v20; // r14
  char v21; // r12
  char *v22; // rbx
  int v23; // edi
  char v24; // r15
  char v25; // si
  size_t v26; // r13
  const char **v27; // rbx
  const char *v28; // r15
  size_t v29; // rdx
  __int64 v30; // r15
  size_t v31; // rax
  size_t v32; // r14
  const char *v33; // r15
  size_t v34; // rax
  const char *v35; // r9
  unsigned int v36; // r13d
  int v37; // esi
  int v38; // eax
  int v39; // edx
  int v40; // ebx
  int v41; // r14d
  unsigned int v42; // r15d
  __int64 v43; // rax
  __int64 v44; // r13
  unsigned int v46; // r15d
  int v47; // eax
  __int64 v48; // rax
  __int64 *v49; // rax
  __int64 v50; // rdx
  int v51; // edx
  __int64 v52; // rax
  const char *v53; // [rsp+8h] [rbp-A8h]
  unsigned int v54; // [rsp+14h] [rbp-9Ch]
  const char *v56; // [rsp+20h] [rbp-90h]
  __int64 v60; // [rsp+38h] [rbp-78h]
  char s1; // [rsp+50h] [rbp-60h]
  __int64 v63; // [rsp+58h] [rbp-58h]
  char v64; // [rsp+60h] [rbp-50h]
  __int64 v65; // [rsp+68h] [rbp-48h]
  int v66; // [rsp+68h] [rbp-48h]
  unsigned int v67; // [rsp+68h] [rbp-48h]
  const char *v68; // [rsp+70h] [rbp-40h] BYREF
  size_t v69; // [rsp+78h] [rbp-38h]

  v54 = *a3;
  v5 = (const char *)(*(__int64 (__fastcall **)(const char **))*a2)(a2);
  v6 = v5;
  if ( v5 )
  {
    v7 = strlen(v5);
    if ( v7 == 1 && *v6 == 45 )
    {
      v8 = sub_1691920((_QWORD *)a1, *(_DWORD *)(a1 + 28));
      v10 = v9;
      goto LABEL_5;
    }
  }
  else
  {
    v7 = 0;
  }
  v12 = *(_DWORD *)(a1 + 48);
  if ( !v12 )
  {
LABEL_79:
    v11 = 0;
    v8 = sub_1691920((_QWORD *)a1, *(_DWORD *)(a1 + 28));
    v10 = v51;
    if ( !v6 )
    {
LABEL_80:
      v67 = (*a3)++;
      v52 = sub_22077B0(80);
      v44 = v52;
      if ( v52 )
        sub_1693430(v52, v8, v10, (_DWORD)v6, v11, v67, (__int64)v6, 0);
      return v44;
    }
LABEL_5:
    v11 = strlen(v6);
    goto LABEL_80;
  }
  v13 = *(__int64 **)(a1 + 40);
  v14 = *v13;
  v15 = v13;
  if ( *v13 )
    goto LABEL_10;
  do
  {
    do
    {
      v14 = v15[1];
      ++v15;
    }
    while ( !v14 );
LABEL_10:
    ;
  }
  while ( v14 == -8 );
  v16 = &v13[v12];
  while ( 1 )
  {
    if ( v16 == v15 )
      goto LABEL_79;
    v17 = *(_QWORD *)*v15;
    if ( v17 > v7 )
      goto LABEL_70;
    if ( !v17 )
      break;
    if ( !memcmp(v6, (const void *)(*v15 + 16), v17) )
    {
      v68 = v6;
      v65 = *(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 36) << 6);
      v60 = *(_QWORD *)(a1 + 8);
LABEL_17:
      v17 = strlen(v6);
      goto LABEL_18;
    }
LABEL_70:
    v48 = v15[1];
    if ( v48 && v48 != -8 )
    {
      ++v15;
    }
    else
    {
      v49 = v15 + 2;
      do
      {
        do
        {
          v50 = *v49;
          v15 = v49++;
        }
        while ( v50 == -8 );
      }
      while ( !v50 );
    }
  }
  v68 = v6;
  v65 = *(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 36) << 6);
  v60 = *(_QWORD *)(a1 + 8);
  if ( v6 )
    goto LABEL_17;
LABEL_18:
  v69 = v17;
  v18 = sub_16D24E0(&v68, *(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80), 0);
  if ( v69 <= v18 )
    v18 = v69;
  v56 = &v68[v18];
  if ( v60 - v65 <= 0 )
    goto LABEL_33;
  v19 = (v60 - v65) >> 6;
  v53 = v6;
  s1 = tolower(*v56);
  while ( 2 )
  {
    v63 = v65 + (v19 >> 1 << 6);
    v20 = *(char **)(v63 + 8);
    v21 = tolower(*v20);
    if ( v21 != s1 )
    {
      v25 = s1;
      goto LABEL_27;
    }
    if ( !s1 )
      goto LABEL_68;
    v22 = (char *)v56;
    while ( 1 )
    {
      v23 = v20[1];
      ++v22;
      ++v20;
      v24 = tolower(v23);
      v21 = v24;
      v25 = tolower(*v22);
      if ( v24 != v25 )
        break;
      if ( !v24 )
        goto LABEL_68;
    }
LABEL_27:
    if ( v21 && (!v25 || v25 > v21) )
    {
      v19 = v19 - (v19 >> 1) - 1;
      v65 = v63 + 64;
    }
    else
    {
LABEL_68:
      v19 >>= 1;
    }
    if ( v19 > 0 )
      continue;
    break;
  }
  v6 = v53;
LABEL_33:
  if ( v65 == v60 )
    goto LABEL_53;
  while ( 2 )
  {
    v26 = 0;
    v64 = *(_BYTE *)(a1 + 24);
    if ( v6 )
      v26 = strlen(v6);
    v27 = *(const char ***)v65;
    if ( !**(_QWORD **)v65 )
      goto LABEL_52;
    v28 = **(const char ***)v65;
    while ( 2 )
    {
      v31 = strlen(v28);
      v32 = v31;
      if ( v26 < v31 || v31 && memcmp(v6, v28, v31) )
        goto LABEL_41;
      v68 = &v6[v32];
      v69 = v26 - v32;
      if ( !v64 )
      {
        v33 = *(const char **)(v65 + 8);
        if ( !v33 )
          goto LABEL_58;
        v34 = strlen(*(const char **)(v65 + 8));
        if ( v26 - v32 >= v34 )
        {
          if ( !v34 )
          {
LABEL_58:
            v35 = v33;
            v46 = v32;
            goto LABEL_59;
          }
          if ( !memcmp(&v6[v32], v33, v34) )
          {
            v35 = v33;
            v36 = v32;
            goto LABEL_51;
          }
        }
        goto LABEL_41;
      }
      v29 = 0;
      v30 = *(_QWORD *)(v65 + 8);
      if ( v30 )
        v29 = strlen(*(const char **)(v65 + 8));
      if ( !(unsigned __int8)sub_16D2000(&v68, v30, v29) )
      {
LABEL_41:
        v28 = v27[1];
        ++v27;
        if ( !v28 )
          goto LABEL_52;
        continue;
      }
      break;
    }
    v46 = v32;
    v35 = *(const char **)(v65 + 8);
LABEL_59:
    v36 = v46;
    if ( v35 )
    {
LABEL_51:
      v36 += strlen(v35);
      if ( !v36 )
        goto LABEL_52;
      goto LABEL_61;
    }
    if ( !v46 )
      goto LABEL_52;
LABEL_61:
    if ( v65 != v60 )
    {
      sub_1690400(&v68, v65, a1);
      v47 = *((unsigned __int16 *)v68 + 19);
      if ( (!a4 || (v47 & a4) != 0) && (v47 & a5) == 0 )
      {
        v44 = sub_16904A0((size_t *)&v68, a2, a3, v36);
        if ( v44 || *a3 != v54 )
          return v44;
      }
LABEL_52:
      v65 += 64;
      if ( v65 != v60 )
        continue;
    }
    break;
  }
LABEL_53:
  if ( *v6 == 47 )
    v37 = *(_DWORD *)(a1 + 28);
  else
    v37 = *(_DWORD *)(a1 + 32);
  v38 = sub_1691920((_QWORD *)a1, v37);
  v40 = v39;
  v66 = v38;
  v41 = strlen(v6);
  v42 = (*a3)++;
  v43 = sub_22077B0(80);
  v44 = v43;
  if ( v43 )
    sub_1693430(v43, v66, v40, (_DWORD)v6, v41, v42, (__int64)v6, 0);
  return v44;
}
