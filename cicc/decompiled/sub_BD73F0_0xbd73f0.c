// Function: sub_BD73F0
// Address: 0xbd73f0
//
unsigned __int64 __fastcall sub_BD73F0(__int64 a1)
{
  __int64 *v2; // rax
  __int64 v3; // r13
  __int64 v4; // rbx
  unsigned int v5; // r8d
  unsigned __int64 v6; // r14
  __int64 v7; // rdi
  int v8; // r11d
  _QWORD *v9; // rax
  unsigned int v10; // edx
  _QWORD *v11; // rsi
  __int64 v12; // r10
  unsigned __int64 v13; // rsi
  _QWORD *v14; // rdx
  unsigned __int64 result; // rax
  unsigned int v16; // r8d
  __int64 v17; // rdi
  int v18; // r15d
  __int64 v19; // rcx
  _QWORD *v20; // rax
  unsigned int v21; // edx
  _QWORD *v22; // rsi
  __int64 v23; // r11
  unsigned __int64 v24; // rsi
  _QWORD *v25; // rcx
  __int64 *v26; // rsi
  unsigned __int64 v27; // rdi
  int v28; // edi
  int v29; // edi
  __int64 v30; // r9
  unsigned int v31; // edx
  int v32; // ecx
  __int64 v33; // r8
  int v34; // esi
  int v35; // edi
  int v36; // edi
  __int64 v37; // r8
  int v38; // edx
  _QWORD *v39; // r9
  __int64 v40; // r15
  __int64 v41; // rsi
  int v42; // esi
  int v43; // edx
  int v44; // r8d
  int v45; // r8d
  int v46; // esi
  __int64 v47; // r9
  _QWORD *v48; // rcx
  unsigned int v49; // edx
  __int64 v50; // rdi
  int v51; // edi
  int v52; // edi
  __int64 v53; // r8
  __int64 v54; // r14
  __int64 v55; // rcx
  int v56; // edx
  _QWORD *v57; // rsi
  int v58; // esi
  _QWORD *v59; // r10

  v2 = (__int64 *)sub_BD5C60(*(_QWORD *)(a1 + 16));
  v3 = *(_QWORD *)(a1 + 16);
  v4 = *v2;
  if ( (*(_BYTE *)(v3 + 1) & 1) != 0 )
  {
    v16 = *(_DWORD *)(v4 + 3192);
    v17 = v4 + 3168;
    if ( v16 )
    {
      v18 = 1;
      v19 = *(_QWORD *)(v4 + 3176);
      v20 = 0;
      v21 = (v16 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v22 = (_QWORD *)(v19 + 16LL * v21);
      v23 = *v22;
      if ( v3 == *v22 )
      {
LABEL_11:
        v24 = (unsigned __int64)(v22 + 1);
        return sub_BD6050((unsigned __int64 *)a1, v24);
      }
      while ( v23 != -4096 )
      {
        if ( !v20 && v23 == -8192 )
          v20 = v22;
        v21 = (v16 - 1) & (v18 + v21);
        v22 = (_QWORD *)(v19 + 16LL * v21);
        v23 = *v22;
        if ( v3 == *v22 )
          goto LABEL_11;
        ++v18;
      }
      if ( !v20 )
        v20 = v22;
      v42 = *(_DWORD *)(v4 + 3184);
      ++*(_QWORD *)(v4 + 3168);
      v43 = v42 + 1;
      if ( 4 * (v42 + 1) < 3 * v16 )
      {
        if ( v16 - *(_DWORD *)(v4 + 3188) - v43 > v16 >> 3 )
        {
LABEL_56:
          *(_DWORD *)(v4 + 3184) = v43;
          if ( *v20 != -4096 )
            --*(_DWORD *)(v4 + 3188);
          *v20 = v3;
          v24 = (unsigned __int64)(v20 + 1);
          v20[1] = 0;
          return sub_BD6050((unsigned __int64 *)a1, v24);
        }
        sub_BD6D00(v17, v16);
        v51 = *(_DWORD *)(v4 + 3192);
        if ( !v51 )
          goto LABEL_97;
        v52 = v51 - 1;
        v53 = *(_QWORD *)(v4 + 3176);
        LODWORD(v54) = v52 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
        v20 = (_QWORD *)(v53 + 16LL * (unsigned int)v54);
        v55 = *v20;
        if ( v3 != *v20 )
        {
          v56 = 1;
          v57 = 0;
          while ( v55 != -4096 )
          {
            if ( v55 == -8192 && !v57 )
              v57 = v20;
            v54 = v52 & (unsigned int)(v54 + v56);
            v20 = (_QWORD *)(v53 + 16 * v54);
            v55 = *v20;
            if ( v3 == *v20 )
              goto LABEL_62;
            ++v56;
          }
          if ( v57 )
          {
            v43 = *(_DWORD *)(v4 + 3184) + 1;
            v20 = v57;
            goto LABEL_56;
          }
        }
LABEL_62:
        v43 = *(_DWORD *)(v4 + 3184) + 1;
        goto LABEL_56;
      }
    }
    else
    {
      ++*(_QWORD *)(v4 + 3168);
    }
    sub_BD6D00(v17, 2 * v16);
    v44 = *(_DWORD *)(v4 + 3192);
    if ( !v44 )
      goto LABEL_97;
    v45 = v44 - 1;
    v46 = 1;
    v47 = *(_QWORD *)(v4 + 3176);
    v48 = 0;
    v49 = v45 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v20 = (_QWORD *)(v47 + 16LL * v49);
    v50 = *v20;
    if ( v3 != *v20 )
    {
      while ( v50 != -4096 )
      {
        if ( !v48 && v50 == -8192 )
          v48 = v20;
        v49 = v45 & (v46 + v49);
        v20 = (_QWORD *)(v47 + 16LL * v49);
        v50 = *v20;
        if ( v3 == *v20 )
          goto LABEL_62;
        ++v46;
      }
      if ( v48 )
      {
        v43 = *(_DWORD *)(v4 + 3184) + 1;
        v20 = v48;
        goto LABEL_56;
      }
    }
    goto LABEL_62;
  }
  v5 = *(_DWORD *)(v4 + 3192);
  v6 = *(_QWORD *)(v4 + 3176);
  v7 = v4 + 3168;
  if ( !v5 )
  {
    ++*(_QWORD *)(v4 + 3168);
    goto LABEL_25;
  }
  v8 = 1;
  v9 = 0;
  v10 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v11 = (_QWORD *)(v6 + 16LL * v10);
  v12 = *v11;
  if ( *v11 == v3 )
  {
LABEL_4:
    v13 = (unsigned __int64)(v11 + 1);
    goto LABEL_5;
  }
  while ( v12 != -4096 )
  {
    if ( !v9 && v12 == -8192 )
      v9 = v11;
    v10 = (v5 - 1) & (v8 + v10);
    v11 = (_QWORD *)(v6 + 16LL * v10);
    v12 = *v11;
    if ( v3 == *v11 )
      goto LABEL_4;
    ++v8;
  }
  if ( !v9 )
    v9 = v11;
  v34 = *(_DWORD *)(v4 + 3184);
  ++*(_QWORD *)(v4 + 3168);
  v32 = v34 + 1;
  if ( 4 * (v34 + 1) >= 3 * v5 )
  {
LABEL_25:
    sub_BD6D00(v7, 2 * v5);
    v28 = *(_DWORD *)(v4 + 3192);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(v4 + 3176);
      v31 = v29 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v32 = *(_DWORD *)(v4 + 3184) + 1;
      v9 = (_QWORD *)(v30 + 16LL * v31);
      v33 = *v9;
      if ( v3 != *v9 )
      {
        v58 = 1;
        v59 = 0;
        while ( v33 != -4096 )
        {
          if ( v33 == -8192 && !v59 )
            v59 = v9;
          v31 = v29 & (v58 + v31);
          v9 = (_QWORD *)(v30 + 16LL * v31);
          v33 = *v9;
          if ( v3 == *v9 )
            goto LABEL_27;
          ++v58;
        }
        if ( v59 )
          v9 = v59;
      }
      goto LABEL_27;
    }
    goto LABEL_97;
  }
  if ( v5 - *(_DWORD *)(v4 + 3188) - v32 <= v5 >> 3 )
  {
    sub_BD6D00(v7, v5);
    v35 = *(_DWORD *)(v4 + 3192);
    if ( v35 )
    {
      v36 = v35 - 1;
      v37 = *(_QWORD *)(v4 + 3176);
      v38 = 1;
      v39 = 0;
      LODWORD(v40) = v36 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v32 = *(_DWORD *)(v4 + 3184) + 1;
      v9 = (_QWORD *)(v37 + 16LL * (unsigned int)v40);
      v41 = *v9;
      if ( v3 != *v9 )
      {
        while ( v41 != -4096 )
        {
          if ( !v39 && v41 == -8192 )
            v39 = v9;
          v40 = v36 & (unsigned int)(v40 + v38);
          v9 = (_QWORD *)(v37 + 16 * v40);
          v41 = *v9;
          if ( v3 == *v9 )
            goto LABEL_27;
          ++v38;
        }
        if ( v39 )
          v9 = v39;
      }
      goto LABEL_27;
    }
LABEL_97:
    ++*(_DWORD *)(v4 + 3184);
    BUG();
  }
LABEL_27:
  *(_DWORD *)(v4 + 3184) = v32;
  if ( *v9 != -4096 )
    --*(_DWORD *)(v4 + 3188);
  *v9 = v3;
  v13 = (unsigned __int64)(v9 + 1);
  v9[1] = 0;
LABEL_5:
  sub_BD6050((unsigned __int64 *)a1, v13);
  *(_BYTE *)(*(_QWORD *)(a1 + 16) + 1LL) |= 1u;
  v14 = *(_QWORD **)(v4 + 3176);
  if ( (unsigned __int64)v14 > v6 || (result = (unsigned __int64)&v14[2 * *(unsigned int *)(v4 + 3192)], v6 >= result) )
  {
    result = *(unsigned int *)(v4 + 3184);
    if ( (unsigned int)result > 1 )
    {
      v25 = &v14[2 * *(unsigned int *)(v4 + 3192)];
      if ( v14 != v25 )
      {
        while ( 1 )
        {
          result = (unsigned __int64)v14;
          if ( *v14 != -8192 && *v14 != -4096 )
            break;
          v14 += 2;
          if ( v25 == v14 )
            return result;
        }
        while ( v25 != (_QWORD *)result )
        {
          v26 = *(__int64 **)(result + 8);
          v27 = result + 8;
          result += 16LL;
          *v26 = v27 | *v26 & 7;
          if ( (_QWORD *)result == v25 )
            break;
          while ( *(_QWORD *)result == -8192 || *(_QWORD *)result == -4096 )
          {
            result += 16LL;
            if ( v25 == (_QWORD *)result )
              return result;
          }
        }
      }
    }
  }
  return result;
}
