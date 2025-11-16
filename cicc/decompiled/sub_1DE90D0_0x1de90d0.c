// Function: sub_1DE90D0
// Address: 0x1de90d0
//
unsigned __int64 __fastcall sub_1DE90D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v7; // rax
  __int64 v8; // r13
  __int64 v9; // rax
  int v10; // edx
  int v11; // esi
  int v12; // r8d
  __int64 v13; // rdi
  unsigned int v14; // edx
  __int64 v15; // rcx
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 *v19; // rsi
  int v20; // ebx
  __int64 v21; // rcx
  int v22; // r8d
  __int64 v23; // rax
  __int64 *v24; // rdi
  __int64 v25; // rcx
  int v26; // eax
  int v27; // r8d
  __int64 v28; // rdi
  unsigned int v29; // esi
  __int64 *v30; // rax
  _QWORD *v31; // rax
  __int64 *v32; // rax
  __int64 v33; // rbx
  unsigned int v34; // eax
  _QWORD *v35; // rdi
  __int64 v36; // rcx
  _QWORD *v37; // rsi
  __int64 v38; // rdx
  _QWORD *v39; // rax
  _QWORD *v40; // r8
  __int64 v41; // rcx
  __int64 v42; // rbx
  __int64 v43; // r14
  unsigned __int64 result; // rax
  __int64 v45; // rsi
  int v46; // r9d
  unsigned int v47; // edx
  __int64 *v48; // r13
  __int64 v49; // rdi
  __int64 *v50; // r15
  char *v51; // rdx
  char *v52; // rdi
  __int64 v53; // rax
  __int64 v54; // rcx
  char *v55; // rax
  _QWORD *v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // rcx
  _QWORD *v59; // rdx
  unsigned __int64 v60; // rdx
  char *v61; // rcx
  __int64 v62; // rbx
  char *v63; // rsi
  __int64 v64; // rdx
  __int64 v65; // rbx
  char *v66; // rdi
  char *v67; // rdx
  size_t v68; // rdx
  char *v69; // rbx
  unsigned __int64 v70; // r8
  int v71; // r14d
  __int64 v72; // r15
  __int64 v73; // rax
  __int64 v74; // r8
  int v75; // eax
  int v76; // r10d
  unsigned __int64 v77; // rdx
  __int64 v78; // rax
  _QWORD *v79; // rdx
  __int64 v80; // [rsp+0h] [rbp-40h] BYREF
  _QWORD *v81; // [rsp+8h] [rbp-38h] BYREF

  v7 = *(_BYTE **)a1;
  v80 = a2;
  *v7 = 1;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = v80;
  v10 = *(_DWORD *)(v8 + 912);
  if ( !v10 )
    goto LABEL_77;
  v11 = v10 - 1;
  v12 = 1;
  v13 = *(_QWORD *)(v8 + 896);
  v14 = (v10 - 1) & (((unsigned int)v80 >> 9) ^ ((unsigned int)v80 >> 4));
  v15 = *(_QWORD *)(v13 + 16LL * v14);
  if ( v80 != v15 )
  {
    while ( v15 != -8 )
    {
      LODWORD(a6) = v12 + 1;
      v14 = v11 & (v12 + v14);
      v15 = *(_QWORD *)(v13 + 16LL * v14);
      if ( v80 == v15 )
        goto LABEL_3;
      ++v12;
    }
LABEL_77:
    v59 = *(_QWORD **)(a1 + 16);
    if ( v80 == *v59 )
    {
      *v59 = *(_QWORD *)(v80 + 8);
      v9 = v80;
      v8 = *(_QWORD *)(a1 + 8);
      v60 = *(unsigned int *)(v8 + 240);
      if ( !*(_BYTE *)(v80 + 180) )
        goto LABEL_79;
      goto LABEL_99;
    }
LABEL_78:
    v60 = *(unsigned int *)(v8 + 240);
    if ( !*(_BYTE *)(v9 + 180) )
    {
LABEL_79:
      v61 = *(char **)(v8 + 232);
      v62 = 8 * v60;
      goto LABEL_80;
    }
LABEL_99:
    v70 = *(unsigned int *)(v8 + 384);
    v71 = *(_DWORD *)(v8 + 384);
    v62 = 8 * v70;
    if ( v70 <= v60 )
    {
      if ( *(_DWORD *)(v8 + 384) )
      {
        memmove(*(void **)(v8 + 232), *(const void **)(v8 + 376), 8 * v70);
        v9 = v80;
      }
      *(_DWORD *)(v8 + 240) = v71;
      v61 = *(char **)(v8 + 232);
    }
    else
    {
      if ( v70 > *(unsigned int *)(v8 + 244) )
      {
        v77 = *(unsigned int *)(v8 + 384);
        v72 = 0;
        *(_DWORD *)(v8 + 240) = 0;
        sub_16CD150(v8 + 232, (const void *)(v8 + 248), v77, 8, v70, a6);
        v70 = *(unsigned int *)(v8 + 384);
      }
      else
      {
        v72 = 8 * v60;
        if ( v60 )
        {
          memmove(*(void **)(v8 + 232), *(const void **)(v8 + 376), 8 * v60);
          v70 = *(unsigned int *)(v8 + 384);
        }
      }
      v73 = *(_QWORD *)(v8 + 376);
      v74 = 8 * v70;
      v61 = *(char **)(v8 + 232);
      if ( v73 + v72 != v74 + v73 )
      {
        memcpy(&v61[v72], (const void *)(v73 + v72), v74 - v72);
        v61 = *(char **)(v8 + 232);
      }
      *(_DWORD *)(v8 + 240) = v71;
      v9 = v80;
    }
LABEL_80:
    v63 = &v61[v62];
    v64 = v62 >> 3;
    v65 = v62 >> 5;
    if ( v65 )
    {
      v66 = v61;
      while ( *(_QWORD *)v66 != v9 )
      {
        if ( *((_QWORD *)v66 + 1) == v9 )
        {
          v66 += 8;
          break;
        }
        if ( *((_QWORD *)v66 + 2) == v9 )
        {
          v66 += 16;
          break;
        }
        if ( *((_QWORD *)v66 + 3) == v9 )
        {
          v66 += 24;
          break;
        }
        v66 += 32;
        if ( &v61[32 * v65] == v66 )
        {
          v64 = (v63 - v66) >> 3;
          goto LABEL_110;
        }
      }
LABEL_87:
      if ( v63 != v66 )
      {
        v67 = v66 + 8;
        if ( v63 == v66 + 8 )
        {
          v69 = v66;
        }
        else
        {
          do
          {
            if ( *(_QWORD *)v67 != v9 )
            {
              *(_QWORD *)v66 = *(_QWORD *)v67;
              v66 += 8;
            }
            v67 += 8;
          }
          while ( v63 != v67 );
          v61 = *(char **)(v8 + 232);
          v68 = &v61[8 * *(unsigned int *)(v8 + 240)] - v63;
          v69 = &v66[v68];
          if ( v63 != &v61[8 * *(unsigned int *)(v8 + 240)] )
          {
            memmove(v66, v63, v68);
            v61 = *(char **)(v8 + 232);
          }
        }
        goto LABEL_94;
      }
      goto LABEL_113;
    }
    v66 = v61;
LABEL_110:
    if ( v64 != 2 )
    {
      if ( v64 != 3 )
      {
        if ( v64 != 1 )
        {
LABEL_113:
          v69 = v63;
LABEL_94:
          *(_DWORD *)(v8 + 240) = (v69 - v61) >> 3;
          goto LABEL_15;
        }
        goto LABEL_118;
      }
      if ( *(_QWORD *)v66 == v9 )
        goto LABEL_87;
      v66 += 8;
    }
    if ( *(_QWORD *)v66 == v9 )
      goto LABEL_87;
    v66 += 8;
LABEL_118:
    v69 = v63;
    if ( *(_QWORD *)v66 != v9 )
      goto LABEL_94;
    goto LABEL_87;
  }
LABEL_3:
  v16 = sub_1DE4FA0(v8 + 888, &v80);
  v17 = v80;
  v18 = v16[1];
  v19 = *(__int64 **)v18;
  v20 = *(_DWORD *)(v18 + 56);
  v21 = *(_QWORD *)v18 + 8LL * *(unsigned int *)(v18 + 8);
  v22 = *(_DWORD *)(v18 + 8);
  if ( *(_QWORD *)v18 != v21 )
  {
    while ( 1 )
    {
      v23 = *v19;
      v24 = v19++;
      if ( v80 == v23 )
        break;
      if ( (__int64 *)v21 == v19 )
        goto LABEL_10;
    }
    if ( (__int64 *)v21 != v19 )
    {
      memmove(v24, v19, v21 - (_QWORD)v19);
      v22 = *(_DWORD *)(v18 + 8);
      v17 = v80;
    }
    *(_DWORD *)(v18 + 8) = v22 - 1;
  }
LABEL_10:
  v25 = *(_QWORD *)(a1 + 8);
  v26 = *(_DWORD *)(v25 + 912);
  if ( v26 )
  {
    v27 = v26 - 1;
    v28 = *(_QWORD *)(v25 + 896);
    v29 = (v26 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
    v30 = (__int64 *)(v28 + 16LL * v29);
    a6 = *v30;
    if ( *v30 == v17 )
    {
LABEL_12:
      *v30 = -16;
      v17 = v80;
      --*(_DWORD *)(v25 + 904);
      ++*(_DWORD *)(v25 + 908);
    }
    else
    {
      v75 = 1;
      while ( a6 != -8 )
      {
        v76 = v75 + 1;
        v29 = v27 & (v75 + v29);
        v30 = (__int64 *)(v28 + 16LL * v29);
        a6 = *v30;
        if ( *v30 == v17 )
          goto LABEL_12;
        v75 = v76;
      }
    }
  }
  v31 = *(_QWORD **)(a1 + 16);
  if ( *v31 == v17 )
  {
    *v31 = *(_QWORD *)(v17 + 8);
    if ( v20 )
      goto LABEL_15;
    goto LABEL_124;
  }
  if ( !v20 )
  {
LABEL_124:
    v8 = *(_QWORD *)(a1 + 8);
    v9 = v80;
    goto LABEL_78;
  }
LABEL_15:
  v32 = *(__int64 **)(a1 + 24);
  v33 = *v32;
  if ( !*v32 || !(unsigned __int8)sub_1DE9010(*v32, &v80, &v81) )
    goto LABEL_28;
  *v81 = -16;
  v34 = *(_DWORD *)(v33 + 8);
  v35 = *(_QWORD **)(v33 + 144);
  ++*(_DWORD *)(v33 + 12);
  *(_DWORD *)(v33 + 8) = (2 * (v34 >> 1) - 2) | v34 & 1;
  v36 = *(unsigned int *)(v33 + 152);
  v37 = &v35[v36];
  v38 = (8 * v36) >> 3;
  if ( (8 * v36) >> 5 )
  {
    v39 = &v35[4 * ((8 * v36) >> 5)];
    while ( *v35 != v80 )
    {
      if ( v80 == v35[1] )
      {
        v40 = ++v35 + 1;
        goto LABEL_25;
      }
      if ( v80 == v35[2] )
      {
        v35 += 2;
        v40 = v35 + 1;
        goto LABEL_25;
      }
      if ( v80 == v35[3] )
      {
        v35 += 3;
        goto LABEL_24;
      }
      v35 += 4;
      if ( v39 == v35 )
      {
        v38 = v37 - v35;
        goto LABEL_136;
      }
    }
    goto LABEL_24;
  }
LABEL_136:
  if ( v38 == 2 )
  {
    v78 = v80;
    v79 = v35;
LABEL_147:
    v35 = v79 + 1;
    if ( *v79 == v78 )
    {
      v35 = v79;
      v40 = v79 + 1;
      goto LABEL_25;
    }
    goto LABEL_144;
  }
  if ( v38 == 3 )
  {
    v40 = v35 + 1;
    v78 = v80;
    v79 = v35 + 1;
    if ( *v35 == v80 )
      goto LABEL_25;
    goto LABEL_147;
  }
  if ( v38 != 1 )
  {
LABEL_139:
    v35 = v37;
    v40 = v37 + 1;
    goto LABEL_25;
  }
  v78 = v80;
LABEL_144:
  if ( *v35 != v78 )
    goto LABEL_139;
LABEL_24:
  v40 = v35 + 1;
LABEL_25:
  if ( v40 != v37 )
  {
    memmove(v35, v40, (char *)v37 - (char *)v40);
    LODWORD(v36) = *(_DWORD *)(v33 + 152);
  }
  *(_DWORD *)(v33 + 152) = v36 - 1;
LABEL_28:
  v41 = *(_QWORD *)(a1 + 8);
  v42 = v80;
  v43 = *(_QWORD *)(v41 + 576);
  result = *(unsigned int *)(v43 + 256);
  if ( (_DWORD)result )
  {
    v45 = *(_QWORD *)(v43 + 240);
    v46 = 1;
    v47 = (result - 1) & (((unsigned int)v80 >> 9) ^ ((unsigned int)v80 >> 4));
    v48 = (__int64 *)(v45 + 16LL * v47);
    v49 = *v48;
    if ( v80 == *v48 )
    {
LABEL_30:
      result = v45 + 16 * result;
      if ( v48 != (__int64 *)result )
      {
        v50 = (__int64 *)v48[1];
        if ( v50 )
        {
          while ( 1 )
          {
            v51 = (char *)v50[5];
            v52 = (char *)v50[4];
            v53 = (v51 - v52) >> 5;
            v54 = (v51 - v52) >> 3;
            if ( v53 <= 0 )
              break;
            v55 = &v52[32 * v53];
            while ( v42 != *(_QWORD *)v52 )
            {
              if ( v42 == *((_QWORD *)v52 + 1) )
              {
                v52 += 8;
                break;
              }
              if ( v42 == *((_QWORD *)v52 + 2) )
              {
                v52 += 16;
                break;
              }
              if ( v42 == *((_QWORD *)v52 + 3) )
              {
                v52 += 24;
                break;
              }
              v52 += 32;
              if ( v52 == v55 )
              {
                v54 = (v51 - v52) >> 3;
                goto LABEL_59;
              }
            }
LABEL_39:
            if ( v52 + 8 != v51 )
            {
              memmove(v52, v52 + 8, v51 - (v52 + 8));
              v51 = (char *)v50[5];
            }
            result = v50[8];
            v50[5] = (__int64)(v51 - 8);
            if ( v50[9] == result )
            {
              v56 = (_QWORD *)(result + 8LL * *((unsigned int *)v50 + 21));
              if ( (_QWORD *)result == v56 )
              {
LABEL_57:
                result = (unsigned __int64)v56;
              }
              else
              {
                while ( v42 != *(_QWORD *)result )
                {
                  result += 8LL;
                  if ( v56 == (_QWORD *)result )
                    goto LABEL_57;
                }
              }
              goto LABEL_52;
            }
            result = (unsigned __int64)sub_16CC9F0((__int64)(v50 + 7), v42);
            if ( v42 == *(_QWORD *)result )
            {
              v57 = v50[9];
              if ( v57 == v50[8] )
                v58 = *((unsigned int *)v50 + 21);
              else
                v58 = *((unsigned int *)v50 + 20);
              v56 = (_QWORD *)(v57 + 8 * v58);
LABEL_52:
              if ( (_QWORD *)result != v56 )
              {
                *(_QWORD *)result = -2;
                ++*((_DWORD *)v50 + 22);
              }
              goto LABEL_44;
            }
            result = v50[9];
            if ( result == v50[8] )
            {
              result += 8LL * *((unsigned int *)v50 + 21);
              v56 = (_QWORD *)result;
              goto LABEL_52;
            }
LABEL_44:
            v50 = (__int64 *)*v50;
            if ( !v50 )
              goto LABEL_45;
          }
LABEL_59:
          if ( v54 != 2 )
          {
            if ( v54 != 3 )
            {
              if ( v54 != 1 )
              {
                v52 = (char *)v50[5];
                goto LABEL_39;
              }
LABEL_71:
              if ( v42 != *(_QWORD *)v52 )
                v52 = (char *)v50[5];
              goto LABEL_39;
            }
            if ( v42 == *(_QWORD *)v52 )
              goto LABEL_39;
            v52 += 8;
          }
          if ( v42 == *(_QWORD *)v52 )
            goto LABEL_39;
          v52 += 8;
          goto LABEL_71;
        }
LABEL_45:
        *v48 = -16;
        v42 = v80;
        --*(_DWORD *)(v43 + 248);
        ++*(_DWORD *)(v43 + 252);
        v41 = *(_QWORD *)(a1 + 8);
      }
    }
    else
    {
      while ( v49 != -8 )
      {
        v47 = (result - 1) & (v46 + v47);
        v48 = (__int64 *)(v45 + 16LL * v47);
        v49 = *v48;
        if ( v80 == *v48 )
          goto LABEL_30;
        ++v46;
      }
    }
  }
  if ( *(_QWORD *)(v41 + 584) == v42 )
    *(_QWORD *)(v41 + 584) = 0;
  return result;
}
