// Function: sub_1DF4120
// Address: 0x1df4120
//
__int64 __fastcall sub_1DF4120(__int64 a1, unsigned int a2)
{
  _QWORD *v2; // r15
  __int64 v3; // r8
  __int64 v4; // rdi
  __int64 v5; // rdx
  int v6; // r9d
  int v7; // r12d
  unsigned int v8; // ecx
  __int64 result; // rax
  _WORD *v10; // rcx
  unsigned __int16 *v11; // r10
  unsigned __int16 v12; // r14
  unsigned __int16 *v13; // rsi
  unsigned __int16 *v14; // r13
  unsigned __int16 *v15; // rax
  int v16; // ecx
  unsigned __int16 *v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rdi
  unsigned int v20; // r9d
  int *v21; // rdx
  int v22; // r10d
  __int64 v23; // r10
  int v24; // esi
  __int64 v25; // rdi
  unsigned int v26; // eax
  __int64 *v27; // r9
  __int64 v28; // r11
  unsigned int v29; // eax
  char *v30; // rdi
  __int64 v31; // rsi
  char *v32; // r9
  __int64 v33; // r10
  __int64 v34; // r10
  char *v35; // rax
  char *v36; // r10
  unsigned __int16 *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  int v40; // esi
  int v41; // edx
  int v42; // r9d
  __int64 v43; // rax
  char *v44; // rdx
  __int64 v45; // [rsp+0h] [rbp-40h]
  int v46; // [rsp+Ch] [rbp-34h]
  int v47; // [rsp+Ch] [rbp-34h]
  int v48; // [rsp+Ch] [rbp-34h]

  v2 = *(_QWORD **)(a1 + 232);
  if ( !v2 )
    BUG();
  v3 = a1;
  v4 = v2[1];
  v5 = v2[7];
  v6 = 0;
  v7 = 0;
  v8 = *(_DWORD *)(v4 + 24LL * a2 + 16);
  result = v8 & 0xF;
  v10 = (_WORD *)(v5 + 2LL * (v8 >> 4));
  v11 = v10 + 1;
  v12 = *v10 + a2 * result;
LABEL_3:
  v13 = v11;
  while ( 1 )
  {
    v14 = v13;
    if ( !v13 )
      break;
    v15 = (unsigned __int16 *)(v2[6] + 4LL * v12);
    v16 = *v15;
    v7 = v15[1];
    if ( (_WORD)v16 )
    {
      while ( 1 )
      {
        result = *(unsigned int *)(v4 + 24LL * (unsigned __int16)v16 + 8);
        v17 = (unsigned __int16 *)(v5 + 2 * result);
        if ( v17 )
          goto LABEL_7;
        if ( !(_WORD)v7 )
          break;
        v16 = v7;
        v7 = 0;
      }
      v6 = v16;
    }
    result = *v13;
    v11 = 0;
    ++v13;
    if ( !(_WORD)result )
      goto LABEL_3;
    v12 += result;
  }
  v16 = v6;
  v17 = 0;
LABEL_7:
  while ( v14 )
  {
    v18 = *(unsigned int *)(v3 + 472);
    if ( !(_DWORD)v18 )
      goto LABEL_25;
    v19 = *(_QWORD *)(v3 + 456);
    v20 = (v18 - 1) & (37 * (unsigned __int16)v16);
    v21 = (int *)(v19 + 16LL * v20);
    v22 = *v21;
    if ( (unsigned __int16)v16 == *v21 )
    {
LABEL_10:
      if ( v21 == (int *)(v19 + 16 * v18) )
        goto LABEL_25;
      if ( (*(_BYTE *)(v3 + 264) & 1) != 0 )
      {
        v23 = v3 + 272;
        v24 = 7;
      }
      else
      {
        v40 = *(_DWORD *)(v3 + 280);
        v23 = *(_QWORD *)(v3 + 272);
        if ( !v40 )
          goto LABEL_25;
        v24 = v40 - 1;
      }
      v25 = *((_QWORD *)v21 + 1);
      v26 = v24 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
      v27 = (__int64 *)(v23 + 8LL * v26);
      v28 = *v27;
      if ( v25 == *v27 )
      {
LABEL_14:
        *v27 = -16;
        v29 = *(_DWORD *)(v3 + 264);
        v30 = *(char **)(v3 + 336);
        ++*(_DWORD *)(v3 + 268);
        *(_DWORD *)(v3 + 264) = (2 * (v29 >> 1) - 2) | v29 & 1;
        v31 = *(unsigned int *)(v3 + 344);
        v32 = &v30[8 * v31];
        v33 = (8 * v31) >> 3;
        if ( (8 * v31) >> 5 )
        {
          v34 = *((_QWORD *)v21 + 1);
          v35 = &v30[32 * ((8 * v31) >> 5)];
          while ( *(_QWORD *)v30 != v34 )
          {
            if ( v34 == *((_QWORD *)v30 + 1) )
            {
              v30 += 8;
              v36 = v30 + 8;
              goto LABEL_22;
            }
            if ( v34 == *((_QWORD *)v30 + 2) )
            {
              v30 += 16;
              v36 = v30 + 8;
              goto LABEL_22;
            }
            if ( v34 == *((_QWORD *)v30 + 3) )
            {
              v30 += 24;
              break;
            }
            v30 += 32;
            if ( v35 == v30 )
            {
              v33 = (v32 - v30) >> 3;
              goto LABEL_43;
            }
          }
LABEL_21:
          v36 = v30 + 8;
LABEL_22:
          if ( v36 != v32 )
          {
            v45 = v3;
            v46 = v16;
            memmove(v30, v36, v32 - v36);
            v3 = v45;
            v16 = v46;
            LODWORD(v31) = *(_DWORD *)(v45 + 344);
          }
          *(_DWORD *)(v3 + 344) = v31 - 1;
          goto LABEL_25;
        }
LABEL_43:
        switch ( v33 )
        {
          case 2LL:
            v43 = *((_QWORD *)v21 + 1);
            v44 = v30;
            break;
          case 3LL:
            v36 = v30 + 8;
            v43 = *((_QWORD *)v21 + 1);
            v44 = v30 + 8;
            if ( *(_QWORD *)v30 == v43 )
              goto LABEL_22;
            break;
          case 1LL:
            v43 = *((_QWORD *)v21 + 1);
LABEL_52:
            if ( v43 == *(_QWORD *)v30 )
              goto LABEL_21;
            goto LABEL_46;
          default:
LABEL_46:
            v30 = v32;
            v36 = v32 + 8;
            goto LABEL_22;
        }
        v30 = v44 + 8;
        if ( v43 == *(_QWORD *)v44 )
        {
          v30 = v44;
          v36 = v44 + 8;
          goto LABEL_22;
        }
        goto LABEL_52;
      }
      v42 = 1;
      while ( v28 != -8 )
      {
        v26 = v24 & (v42 + v26);
        v48 = v42 + 1;
        v27 = (__int64 *)(v23 + 8LL * v26);
        v28 = *v27;
        if ( v25 == *v27 )
          goto LABEL_14;
        v42 = v48;
      }
    }
    else
    {
      v41 = 1;
      while ( v22 != -1 )
      {
        v20 = (v18 - 1) & (v41 + v20);
        v47 = v41 + 1;
        v21 = (int *)(v19 + 16LL * v20);
        v22 = *v21;
        if ( (unsigned __int16)v16 == *v21 )
          goto LABEL_10;
        v41 = v47;
      }
    }
LABEL_25:
    result = *v17++;
    v16 += result;
    if ( (_WORD)result )
      continue;
    if ( (_WORD)v7 )
    {
      v38 = v2[1] + 24LL * (unsigned __int16)v7;
      v16 = v7;
      v7 = 0;
      v39 = *(unsigned int *)(v38 + 8);
      result = v2[7];
      v17 = (unsigned __int16 *)(result + 2 * v39);
      continue;
    }
    v7 = *v14;
    v12 += v7;
    if ( !(_WORD)v7 )
    {
      v17 = 0;
      v14 = 0;
      continue;
    }
    ++v14;
    v37 = (unsigned __int16 *)(v2[6] + 4LL * v12);
    v16 = *v37;
    v7 = v37[1];
    result = v2[7];
    v17 = (unsigned __int16 *)(result + 2LL * *(unsigned int *)(v2[1] + 24LL * (unsigned __int16)v16 + 8));
  }
  return result;
}
