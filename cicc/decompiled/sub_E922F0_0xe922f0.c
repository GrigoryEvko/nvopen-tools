// Function: sub_E922F0
// Address: 0xe922f0
//
char *__fastcall sub_E922F0(_QWORD *a1, unsigned int a2)
{
  __int64 v2; // rdx
  __int16 v3; // r14
  __int64 v4; // r8
  char *v5; // r13
  char *result; // rax
  _QWORD *v7; // rcx
  __int64 v8; // rdi
  __int64 v9; // r11
  __int64 v10; // r10
  __int16 *v11; // r9
  unsigned __int16 *v12; // rdx
  unsigned __int16 v13; // ax
  unsigned __int16 v14; // r15
  int v15; // r12d
  __int16 *v16; // rdx
  __int16 *v17; // rbx
  int v18; // eax
  int v19; // eax
  unsigned __int16 *v20; // rdx
  int v21; // edx
  unsigned __int16 *v22; // r12
  unsigned __int64 v23; // rax
  char *v24; // rbx
  __int16 v25; // cx
  char *v26; // rax
  __int16 v27; // ax
  char *v28; // rbx
  unsigned __int16 v29; // cx
  unsigned __int16 v30; // dx
  char *v31; // rax
  char *v32; // rsi
  char *v33; // rdx
  int v34; // edx
  _QWORD *v35; // [rsp+0h] [rbp-60h]
  int v36; // [rsp+Ch] [rbp-54h]
  __int16 *v37; // [rsp+10h] [rbp-50h]
  __int16 *v38; // [rsp+18h] [rbp-48h]
  __int64 v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+18h] [rbp-48h]
  __int64 v41; // [rsp+18h] [rbp-48h]
  __int64 v42; // [rsp+18h] [rbp-48h]
  _WORD v43[25]; // [rsp+2Eh] [rbp-32h] BYREF

  v2 = 24LL * a2;
  v3 = a2;
  v4 = v2 + a1[28];
  v5 = *(char **)(v4 + 8);
  result = *(char **)v4;
  if ( *(char **)v4 != v5 )
    return result;
  v7 = a1;
  v8 = a1[1];
  v9 = v7[7];
  LODWORD(v10) = *(_DWORD *)(v8 + v2 + 16) & 0xFFF;
  if ( !(v9 + 2LL * (*(_DWORD *)(v8 + v2 + 16) >> 12)) )
    goto LABEL_28;
  v38 = (__int16 *)(v9 + 2LL * (*(_DWORD *)(v8 + v2 + 16) >> 12));
  while ( 1 )
  {
    v11 = v38;
    v12 = (unsigned __int16 *)(v7[6] + 4LL * (unsigned int)v10);
    v13 = *v12;
    v14 = v12[1];
    if ( *v12 )
      break;
LABEL_59:
    v34 = *v38++;
    LODWORD(v10) = v34 + v10;
    if ( !(_WORD)v34 )
      goto LABEL_28;
  }
  while ( 1 )
  {
    v15 = v13;
    v16 = (__int16 *)(v9 + 2LL * *(unsigned int *)(v8 + 24LL * v13 + 8));
    if ( !v16 )
      goto LABEL_32;
    v17 = (__int16 *)(v9 + 2LL * *(unsigned int *)(v8 + 24LL * v13 + 8));
    if ( a2 != v13 )
      break;
    while ( 1 )
    {
      v18 = *v16++;
      if ( !(_WORD)v18 )
        break;
      v15 += v18;
      v17 = v16;
      v13 = v15;
      if ( a2 != (unsigned __int16)v15 )
        goto LABEL_9;
    }
LABEL_32:
    v13 = v14;
    if ( !v14 )
      goto LABEL_59;
    v14 = 0;
  }
LABEL_9:
  if ( !v38 )
    goto LABEL_28;
  while ( 2 )
  {
    v43[0] = v13;
    if ( *(char **)(v4 + 16) == v5 )
    {
      v35 = v7;
      v36 = v10;
      v37 = v11;
      v40 = v4;
      sub_C8FDD0(v4, v5, v43);
      v4 = v40;
      v7 = v35;
      LODWORD(v10) = v36;
      v11 = v37;
      v5 = *(char **)(v40 + 8);
    }
    else
    {
      if ( v5 )
      {
        *(_WORD *)v5 = v13;
        v5 = *(char **)(v4 + 8);
      }
      v5 += 2;
      *(_QWORD *)(v4 + 8) = v5;
    }
LABEL_18:
    if ( *v17 )
    {
      v15 += *v17++;
      v13 = v15;
      goto LABEL_20;
    }
    if ( v14 )
    {
      v15 = v14;
      v17 = (__int16 *)(v7[7] + 2LL * *(unsigned int *)(v7[1] + 24LL * v14 + 8));
      v13 = v14;
      v14 = 0;
LABEL_20:
      v21 = v13;
LABEL_17:
      if ( a2 != v21 )
        continue;
      goto LABEL_18;
    }
    break;
  }
  v19 = *v11;
  if ( *v11 )
  {
    ++v11;
    v10 = (unsigned int)(v10 + v19);
    v20 = (unsigned __int16 *)(v7[6] + 4 * v10);
    v15 = *v20;
    v14 = v20[1];
    v13 = *v20;
    v17 = (__int16 *)(v7[7] + 2LL * *(unsigned int *)(v7[1] + 24LL * (unsigned __int16)v15 + 8));
    v21 = v15;
    goto LABEL_17;
  }
  v22 = *(unsigned __int16 **)v4;
  if ( *(char **)v4 != v5 )
  {
    v39 = v4;
    _BitScanReverse64(&v23, (v5 - (char *)v22) >> 1);
    sub_E91B00(*(char **)v4, v5, 2LL * (int)(63 - (v23 ^ 0x3F)));
    if ( v5 - (char *)v22 > 32 )
    {
      v28 = (char *)(v22 + 16);
      sub_E91A40(v22, v22 + 16);
      v4 = v39;
      if ( v22 + 16 != (unsigned __int16 *)v5 )
      {
        do
        {
          v29 = *(_WORD *)v28;
          v30 = *((_WORD *)v28 - 1);
          v31 = v28 - 2;
          if ( *(_WORD *)v28 >= v30 )
          {
            v32 = v28;
          }
          else
          {
            do
            {
              *((_WORD *)v31 + 1) = v30;
              v32 = v31;
              v30 = *((_WORD *)v31 - 1);
              v31 -= 2;
            }
            while ( v29 < v30 );
          }
          v28 += 2;
          *(_WORD *)v32 = v29;
        }
        while ( v28 != v5 );
      }
    }
    else
    {
      sub_E91A40(v22, (unsigned __int16 *)v5);
      v4 = v39;
    }
    v24 = *(char **)(v4 + 8);
    v5 = *(char **)v4;
    if ( v24 != *(char **)v4 )
    {
      while ( 1 )
      {
        v26 = v5;
        v5 += 2;
        if ( v24 == v5 )
          break;
        v25 = *((_WORD *)v5 - 1);
        if ( v25 == *(_WORD *)v5 )
        {
          if ( v24 == v26 )
          {
            v5 = *(char **)(v4 + 8);
            break;
          }
          v33 = v26 + 4;
          if ( v24 == v26 + 4 )
          {
            if ( v24 == v5 )
            {
              v43[0] = v3;
              v27 = v3;
              if ( *(char **)(v4 + 16) != v5 )
                goto LABEL_57;
              goto LABEL_43;
            }
          }
          else
          {
            while ( 1 )
            {
              if ( v25 != *(_WORD *)v33 )
              {
                *((_WORD *)v26 + 1) = *(_WORD *)v33;
                v26 += 2;
              }
              v33 += 2;
              if ( v24 == v33 )
                break;
              v25 = *(_WORD *)v26;
            }
            v5 = v26 + 2;
            v33 = *(char **)(v4 + 8);
            if ( v24 == v26 + 2 )
            {
              v5 = *(char **)(v4 + 8);
              break;
            }
            if ( v24 != v33 )
            {
              v42 = v4;
              memmove(v26 + 2, v24, v33 - v24);
              v4 = v42;
              v33 = *(char **)(v42 + 8);
            }
          }
          v5 += v33 - v24;
          if ( v33 != v5 )
            *(_QWORD *)(v4 + 8) = v5;
          v43[0] = v3;
          v27 = v3;
          if ( v5 == *(char **)(v4 + 16) )
            goto LABEL_43;
          goto LABEL_57;
        }
      }
    }
  }
LABEL_28:
  v43[0] = v3;
  v27 = v3;
  if ( v5 == *(char **)(v4 + 16) )
  {
LABEL_43:
    v41 = v4;
    sub_C8FDD0(v4, v5, v43);
    v4 = v41;
  }
  else
  {
    if ( v5 )
    {
LABEL_57:
      *(_WORD *)v5 = v27;
      v5 = *(char **)(v4 + 8);
    }
    *(_QWORD *)(v4 + 8) = v5 + 2;
  }
  return *(char **)v4;
}
