// Function: sub_29E69F0
// Address: 0x29e69f0
//
_BYTE *__fastcall sub_29E69F0(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 *v7; // rbx
  _QWORD *v8; // rdi
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rax
  _BYTE *v12; // r13
  int v13; // eax
  unsigned __int8 *v14; // rcx
  unsigned int v15; // eax
  char v16; // r15
  unsigned int v17; // esi
  int v18; // r11d
  __int64 *v19; // rdx
  _QWORD *v20; // rax
  unsigned __int8 *v21; // rdi
  _QWORD *v22; // rax
  int v23; // eax
  __int64 v24; // rdx
  __int64 v25; // r14
  char *v26; // r13
  unsigned __int8 v27; // al
  int v29; // eax
  int v30; // eax
  __int64 v31; // rax
  __int64 v32; // rsi
  unsigned int v33; // ecx
  __int64 *v34; // rdx
  int v35; // eax
  unsigned __int8 *v36; // rdx
  __int64 v37; // rdi
  __int64 *v38; // rax
  int v39; // edi
  int v40; // edi
  unsigned int v41; // esi
  int v42; // r11d
  __int64 *v43; // r10
  __int64 *v44; // r14
  __int64 v45; // rcx
  __int64 v46; // rax
  __int64 v47; // r15
  char *v48; // r13
  char v49; // al
  __int64 v50; // rax
  __int64 v51; // rsi
  unsigned int v52; // ecx
  __int64 *v53; // rdx
  int v54; // edx
  __int64 v55; // rax
  unsigned __int64 v56; // rdx
  int v57; // esi
  int v58; // esi
  unsigned int v59; // r14d
  int v60; // r10d
  unsigned __int8 *v61; // rdi
  int v62; // edx
  __int64 v63; // rax
  unsigned __int64 v64; // rdx
  unsigned __int8 *v65; // [rsp+10h] [rbp-90h]
  __int64 *v66; // [rsp+10h] [rbp-90h]
  unsigned __int8 *v67; // [rsp+10h] [rbp-90h]
  _QWORD *v69; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v70; // [rsp+28h] [rbp-78h]
  unsigned int v71; // [rsp+2Ch] [rbp-74h]
  _QWORD v72[14]; // [rsp+30h] [rbp-70h] BYREF

  v7 = a1;
  v69 = v72;
  v72[0] = a1;
  v8 = v72;
  v9 = 0;
  v71 = 8;
  v70 = 0;
  if ( *v7 != 39 )
    goto LABEL_22;
  while ( 2 )
  {
    v10 = *((_QWORD *)v7 - 1);
    if ( (v7[2] & 1) != 0 )
    {
      v11 = sub_AA4FF0(*(_QWORD *)(v10 + 32));
      if ( v11 )
        goto LABEL_4;
LABEL_29:
      v9 = v70;
      v8 = v69;
      goto LABEL_30;
    }
    v44 = (__int64 *)(v10 + 32);
    v45 = 32LL * (*((_DWORD *)v7 + 1) & 0x7FFFFFF);
    v66 = (__int64 *)(v10 + v45);
    if ( v10 + 32 != v10 + v45 )
    {
      while ( 1 )
      {
        v46 = sub_AA4FF0(*v44);
        if ( !v46 )
          BUG();
        v47 = *(_QWORD *)(v46 - 8);
        v44 += 4;
        if ( v47 )
          break;
LABEL_95:
        if ( v44 == v66 )
          goto LABEL_29;
      }
      while ( 1 )
      {
        v48 = *(char **)(v47 + 24);
        v49 = *v48;
        if ( (unsigned __int8)*v48 <= 0x1Cu || v49 != 39 && v49 != 80 )
          goto LABEL_80;
        v50 = *(unsigned int *)(a2 + 24);
        v51 = *(_QWORD *)(a2 + 8);
        if ( !(_DWORD)v50 )
          goto LABEL_92;
        v52 = (v50 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
        v53 = (__int64 *)(v51 + 16LL * v52);
        a5 = *v53;
        if ( v48 != (char *)*v53 )
        {
          v54 = 1;
          while ( a5 != -4096 )
          {
            a6 = (unsigned int)(v54 + 1);
            v52 = (v50 - 1) & (v54 + v52);
            v53 = (__int64 *)(v51 + 16LL * v52);
            a5 = *v53;
            if ( v48 == (char *)*v53 )
              goto LABEL_85;
            v54 = a6;
          }
          goto LABEL_92;
        }
LABEL_85:
        if ( v53 == (__int64 *)(v51 + 16 * v50) )
        {
LABEL_92:
          v55 = v70;
          v56 = v70 + 1LL;
          if ( v56 > v71 )
          {
            sub_C8D5F0((__int64)&v69, v72, v56, 8u, a5, a6);
            v55 = v70;
          }
          v69[v55] = v48;
          ++v70;
          v47 = *(_QWORD *)(v47 + 8);
          if ( !v47 )
            goto LABEL_95;
        }
        else
        {
          v12 = (_BYTE *)v53[1];
          if ( v12 && *v12 == 21 )
          {
            v14 = 0;
            if ( v44 != v66 )
              goto LABEL_9;
            goto LABEL_89;
          }
LABEL_80:
          v47 = *(_QWORD *)(v47 + 8);
          if ( !v47 )
            goto LABEL_95;
        }
      }
    }
LABEL_20:
    if ( !v9 )
    {
LABEL_31:
      v12 = 0;
      goto LABEL_32;
    }
LABEL_21:
    v24 = v9--;
    v7 = (unsigned __int8 *)v8[v24 - 1];
    v70 = v9;
    if ( *v7 == 39 )
      continue;
    break;
  }
LABEL_22:
  v25 = *((_QWORD *)v7 + 2);
  if ( !v25 )
    goto LABEL_20;
  while ( 1 )
  {
    v26 = *(char **)(v25 + 24);
    v27 = *v26;
    if ( (unsigned __int8)*v26 <= 0x1Cu )
      goto LABEL_28;
    if ( v27 == 37 )
      break;
    if ( v27 == 34 )
    {
      v12 = (_BYTE *)sub_AA4FF0(*((_QWORD *)v26 - 8));
      if ( v12 )
        v12 -= 24;
LABEL_56:
      v35 = (unsigned __int8)*v12;
      if ( (unsigned __int8)v35 <= 0x1Cu )
      {
        v14 = 0;
        goto LABEL_9;
      }
      v15 = v35 - 80;
      if ( v15 <= 1 )
      {
        v36 = (unsigned __int8 *)*((_QWORD *)v12 - 4);
      }
      else
      {
        v36 = (unsigned __int8 *)**((_QWORD **)v12 - 1);
        if ( !v36 )
          goto LABEL_51;
      }
      if ( v36 != v7 )
        goto LABEL_7;
      goto LABEL_28;
    }
    if ( v27 == 39 || v27 == 80 )
    {
      v31 = *(unsigned int *)(a2 + 24);
      v32 = *(_QWORD *)(a2 + 8);
      if ( !(_DWORD)v31 )
        goto LABEL_111;
      v33 = (v31 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v34 = (__int64 *)(v32 + 16LL * v33);
      a5 = *v34;
      if ( v26 != (char *)*v34 )
      {
        v62 = 1;
        while ( a5 != -4096 )
        {
          a6 = (unsigned int)(v62 + 1);
          v33 = (v31 - 1) & (v62 + v33);
          v34 = (__int64 *)(v32 + 16LL * v33);
          a5 = *v34;
          if ( v26 == (char *)*v34 )
            goto LABEL_54;
          v62 = a6;
        }
LABEL_111:
        v63 = v70;
        v64 = v70 + 1LL;
        if ( v64 > v71 )
        {
          sub_C8D5F0((__int64)&v69, v72, v64, 8u, a5, a6);
          v63 = v70;
        }
        v69[v63] = v26;
        ++v70;
        goto LABEL_28;
      }
LABEL_54:
      if ( v34 == (__int64 *)(v32 + 16 * v31) )
        goto LABEL_111;
      v12 = (_BYTE *)v34[1];
      if ( v12 )
        goto LABEL_56;
    }
LABEL_28:
    v25 = *(_QWORD *)(v25 + 8);
    if ( !v25 )
      goto LABEL_29;
  }
  if ( (v26[2] & 1) == 0 || (v37 = *(_QWORD *)&v26[32 * (1LL - (*((_DWORD *)v26 + 1) & 0x7FFFFFF))]) == 0 )
  {
    v38 = (__int64 *)sub_BD5C60((__int64)v7);
    v12 = (_BYTE *)sub_AC3540(v38);
    if ( v12 )
    {
LABEL_89:
      v13 = (unsigned __int8)*v12;
      goto LABEL_5;
    }
    v9 = v70;
    v8 = v69;
LABEL_30:
    if ( !v9 )
      goto LABEL_31;
    goto LABEL_21;
  }
  v11 = sub_AA4FF0(v37);
  if ( !v11 )
  {
    v9 = v70;
    v8 = v69;
    goto LABEL_30;
  }
LABEL_4:
  v12 = (_BYTE *)(v11 - 24);
  v13 = *(unsigned __int8 *)(v11 - 24);
LABEL_5:
  v14 = 0;
  if ( (unsigned __int8)v13 > 0x1Cu )
  {
    v15 = v13 - 80;
LABEL_7:
    if ( v15 > 1 )
LABEL_51:
      v14 = (unsigned __int8 *)**((_QWORD **)v12 - 1);
    else
      v14 = (unsigned __int8 *)*((_QWORD *)v12 - 4);
  }
LABEL_9:
  v16 = 0;
  while ( 2 )
  {
    if ( v14 != v7 )
    {
      if ( *v7 == 81 )
        goto LABEL_49;
      v17 = *(_DWORD *)(a2 + 24);
      if ( v17 )
      {
        a6 = *(_QWORD *)(a2 + 8);
        v18 = 1;
        v19 = 0;
        a5 = (v17 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v20 = (_QWORD *)(a6 + 16 * a5);
        v21 = (unsigned __int8 *)*v20;
        if ( v7 == (unsigned __int8 *)*v20 )
        {
LABEL_14:
          v22 = v20 + 1;
          goto LABEL_15;
        }
        while ( v21 != (unsigned __int8 *)-4096LL )
        {
          if ( v21 == (unsigned __int8 *)-8192LL && !v19 )
            v19 = v20;
          a5 = (v17 - 1) & (v18 + (_DWORD)a5);
          v20 = (_QWORD *)(a6 + 16LL * (unsigned int)a5);
          v21 = (unsigned __int8 *)*v20;
          if ( v7 == (unsigned __int8 *)*v20 )
            goto LABEL_14;
          ++v18;
        }
        if ( !v19 )
          v19 = v20;
        v29 = *(_DWORD *)(a2 + 16);
        ++*(_QWORD *)a2;
        v30 = v29 + 1;
        if ( 4 * v30 < 3 * v17 )
        {
          a5 = v17 >> 3;
          if ( v17 - *(_DWORD *)(a2 + 20) - v30 <= (unsigned int)a5 )
          {
            v67 = v14;
            sub_2685CE0(a2, v17);
            v57 = *(_DWORD *)(a2 + 24);
            if ( !v57 )
            {
LABEL_128:
              ++*(_DWORD *)(a2 + 16);
              BUG();
            }
            v58 = v57 - 1;
            a5 = *(_QWORD *)(a2 + 8);
            a6 = 0;
            v59 = v58 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
            v14 = v67;
            v60 = 1;
            v30 = *(_DWORD *)(a2 + 16) + 1;
            v19 = (__int64 *)(a5 + 16LL * v59);
            v61 = (unsigned __int8 *)*v19;
            if ( v7 != (unsigned __int8 *)*v19 )
            {
              while ( v61 != (unsigned __int8 *)-4096LL )
              {
                if ( !a6 && v61 == (unsigned __int8 *)-8192LL )
                  a6 = (__int64)v19;
                v59 = v58 & (v60 + v59);
                v19 = (__int64 *)(a5 + 16LL * v59);
                v61 = (unsigned __int8 *)*v19;
                if ( v7 == (unsigned __int8 *)*v19 )
                  goto LABEL_45;
                ++v60;
              }
              if ( a6 )
                v19 = (__int64 *)a6;
            }
          }
          goto LABEL_45;
        }
      }
      else
      {
        ++*(_QWORD *)a2;
      }
      v65 = v14;
      sub_2685CE0(a2, 2 * v17);
      v39 = *(_DWORD *)(a2 + 24);
      if ( !v39 )
        goto LABEL_128;
      v40 = v39 - 1;
      a6 = *(_QWORD *)(a2 + 8);
      v14 = v65;
      v41 = v40 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v30 = *(_DWORD *)(a2 + 16) + 1;
      v19 = (__int64 *)(a6 + 16LL * v41);
      a5 = *v19;
      if ( v7 != (unsigned __int8 *)*v19 )
      {
        v42 = 1;
        v43 = 0;
        while ( a5 != -4096 )
        {
          if ( a5 == -8192 && !v43 )
            v43 = v19;
          v41 = v40 & (v42 + v41);
          v19 = (__int64 *)(a6 + 16LL * v41);
          a5 = *v19;
          if ( v7 == (unsigned __int8 *)*v19 )
            goto LABEL_45;
          ++v42;
        }
        if ( v43 )
          v19 = v43;
      }
LABEL_45:
      *(_DWORD *)(a2 + 16) = v30;
      if ( *v19 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v19 = (__int64)v7;
      v22 = v19 + 1;
      v19[1] = 0;
LABEL_15:
      *v22 = v12;
      v16 |= v7 == a1;
      v23 = *v7;
      if ( (unsigned __int8)v23 > 0x1Cu && (unsigned int)(v23 - 80) <= 1 )
      {
LABEL_49:
        v7 = (unsigned __int8 *)*((_QWORD *)v7 - 4);
      }
      else
      {
        v7 = (unsigned __int8 *)**((_QWORD **)v7 - 1);
        if ( !v7 )
          BUG();
      }
      if ( *v7 <= 0x1Cu )
        break;
      continue;
    }
    break;
  }
  v8 = v69;
  if ( !v16 )
  {
    v9 = v70;
    goto LABEL_20;
  }
LABEL_32:
  if ( v8 != v72 )
    _libc_free((unsigned __int64)v8);
  return v12;
}
