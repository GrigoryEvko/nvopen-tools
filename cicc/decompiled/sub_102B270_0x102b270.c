// Function: sub_102B270
// Address: 0x102b270
//
unsigned __int64 __fastcall sub_102B270(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 v5; // rcx
  unsigned int v6; // edx
  __int64 v7; // rsi
  __int64 v8; // r12
  __int64 v9; // r13
  unsigned __int64 v10; // rsi
  __int64 v11; // rax
  _BYTE *v12; // r13
  bool v13; // di
  unsigned __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rcx
  unsigned int v17; // edx
  _QWORD *v18; // r12
  __int64 v19; // r9
  unsigned __int64 v20; // rax
  __int64 v21; // rdx
  unsigned int v22; // ecx
  __int64 v23; // rdi
  unsigned int v24; // edx
  __int64 *v25; // r14
  __int64 v26; // rsi
  _BYTE *v27; // rsi
  _QWORD *v28; // rdx
  int v29; // ecx
  _QWORD *v30; // rax
  __int64 v31; // rcx
  int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rsi
  unsigned int v36; // edx
  __int64 v37; // r12
  _BYTE *v38; // rcx
  __int64 v39; // rsi
  _QWORD *v40; // rax
  _QWORD *v41; // r14
  __int64 v42; // rdx
  _QWORD *v43; // r13
  int v44; // edi
  int v45; // edi
  __int64 v46; // r8
  unsigned int v47; // esi
  _QWORD *v48; // rax
  __int64 v49; // r10
  __int64 v50; // rsi
  _QWORD *v51; // rax
  int v52; // r8d
  int v53; // eax
  int v54; // r9d
  __int64 *v55; // rax
  int v56; // eax
  int v57; // r10d
  int v58; // r8d
  int v59; // r9d
  _QWORD *v60; // [rsp+0h] [rbp-80h]
  __int64 *v61; // [rsp+8h] [rbp-78h]
  _QWORD v62[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v63; // [rsp+20h] [rbp-60h]
  __int64 v64; // [rsp+30h] [rbp-50h] BYREF
  __int64 v65; // [rsp+38h] [rbp-48h]
  unsigned __int64 v66; // [rsp+40h] [rbp-40h]

  if ( !*(_DWORD *)(a1 + 48) )
    goto LABEL_2;
  v64 = 0;
  v12 = (_BYTE *)(a2 & 0xFFFFFFFFFFFFFFF8LL);
  v65 = 0;
  v13 = (a2 & 0xFFFFFFFFFFFFFFF8LL) != 0;
  v66 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (a2 & 0xFFFFFFFFFFFFEFF8LL) == 0xFFFFFFFFFFFFE000LL || (a2 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    v14 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  }
  else
  {
    sub_BD73F0((__int64)&v64);
    v14 = v66;
    v13 = v66 != 0;
  }
  v15 = *(unsigned int *)(a1 + 56);
  v16 = *(_QWORD *)(a1 + 40);
  if ( !(_DWORD)v15 )
  {
LABEL_82:
    v18 = (_QWORD *)(v16 + 48 * v15);
    if ( v14 == -4096 || v14 == -8192 || !v13 )
      goto LABEL_46;
    goto LABEL_26;
  }
  v17 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
  v18 = (_QWORD *)(v16 + 48LL * v17);
  v19 = v18[2];
  if ( v19 != v14 )
  {
    v57 = 1;
    while ( v19 != -4096 )
    {
      v17 = (v15 - 1) & (v57 + v17);
      v18 = (_QWORD *)(v16 + 48LL * v17);
      v19 = v18[2];
      if ( v19 == v14 )
        goto LABEL_24;
      ++v57;
    }
    goto LABEL_82;
  }
LABEL_24:
  if ( v14 != -8192 && v14 != -4096 && v13 )
  {
LABEL_26:
    sub_BD60C0(&v64);
    v15 = *(unsigned int *)(a1 + 56);
    v16 = *(_QWORD *)(a1 + 40);
  }
  if ( v18 == (_QWORD *)(v16 + 48 * v15) )
    goto LABEL_46;
  v20 = v18[4] & 0xFFFFFFFFFFFFFFF8LL;
  v21 = v18[4] & 7LL;
  if ( (unsigned int)v21 > 2 )
  {
    if ( (_DWORD)v21 != 3 )
LABEL_115:
      BUG();
    v20 = 0;
  }
  v22 = *(_DWORD *)(a1 + 88);
  v23 = *(_QWORD *)(a1 + 72);
  if ( v22 )
  {
    v24 = (v22 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
    v25 = (__int64 *)(v23 + 72LL * v24);
    v26 = *v25;
    if ( v20 == *v25 )
      goto LABEL_31;
    v59 = 1;
    while ( v26 != -4096 )
    {
      v24 = (v22 - 1) & (v59 + v24);
      v25 = (__int64 *)(v23 + 72LL * v24);
      v26 = *v25;
      if ( v20 == *v25 )
        goto LABEL_31;
      ++v59;
    }
  }
  v25 = (__int64 *)(v23 + 72LL * v22);
LABEL_31:
  if ( *((_BYTE *)v25 + 36) )
  {
    v27 = (_BYTE *)v25[2];
    v28 = &v27[8 * *((unsigned int *)v25 + 7)];
    v29 = *((_DWORD *)v25 + 7);
    if ( v27 == (_BYTE *)v28 )
    {
LABEL_95:
      if ( v29 != *((_DWORD *)v25 + 8) )
        goto LABEL_38;
LABEL_96:
      *v25 = -8192;
      --*(_DWORD *)(a1 + 80);
      ++*(_DWORD *)(a1 + 84);
      goto LABEL_38;
    }
    v30 = (_QWORD *)v25[2];
    while ( v12 != (_BYTE *)*v30 )
    {
      if ( v28 == ++v30 )
        goto LABEL_95;
    }
    v31 = (unsigned int)(v29 - 1);
    *((_DWORD *)v25 + 7) = v31;
    *v30 = *(_QWORD *)&v27[8 * v31];
    v32 = *((_DWORD *)v25 + 8);
    ++v25[1];
  }
  else
  {
    v27 = v12;
    v55 = sub_C8CA60((__int64)(v25 + 1), (__int64)v12);
    if ( v55 )
    {
      *v55 = -2;
      v56 = *((_DWORD *)v25 + 8);
      ++v25[1];
      v32 = v56 + 1;
      *((_DWORD *)v25 + 8) = v32;
    }
    else
    {
      v32 = *((_DWORD *)v25 + 8);
    }
  }
  if ( *((_DWORD *)v25 + 7) == v32 )
  {
    if ( !*((_BYTE *)v25 + 36) )
      _libc_free(v25[2], v27);
    goto LABEL_96;
  }
LABEL_38:
  v64 = 0;
  v65 = 0;
  v66 = -8192;
  v33 = v18[2];
  if ( v33 != -8192 )
  {
    if ( v33 && v33 != -4096 )
      sub_BD60C0(v18);
    v18[2] = -8192;
    if ( v66 != -4096 && v66 != 0 && v66 != -8192 )
      sub_BD60C0(&v64);
  }
  --*(_DWORD *)(a1 + 48);
  ++*(_DWORD *)(a1 + 52);
LABEL_46:
  if ( *v12 <= 0x1Cu )
    goto LABEL_2;
  v34 = *(unsigned int *)(a1 + 88);
  v35 = *(_QWORD *)(a1 + 72);
  if ( !(_DWORD)v34 )
    goto LABEL_2;
  v36 = (v34 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
  v37 = v35 + 72LL * v36;
  v38 = *(_BYTE **)v37;
  if ( v12 == *(_BYTE **)v37 )
  {
LABEL_49:
    if ( v37 == v35 + 72 * v34 )
      goto LABEL_2;
    v39 = *(unsigned __int8 *)(v37 + 36);
    v40 = *(_QWORD **)(v37 + 16);
    if ( (_BYTE)v39 )
    {
      v41 = &v40[*(unsigned int *)(v37 + 28)];
      if ( v40 != v41 )
        goto LABEL_52;
    }
    else
    {
      v41 = &v40[*(unsigned int *)(v37 + 24)];
      if ( v40 != v41 )
      {
LABEL_52:
        while ( 1 )
        {
          v42 = *v40;
          v43 = v40;
          if ( *v40 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v41 == ++v40 )
            goto LABEL_54;
        }
        if ( v40 != v41 )
        {
          do
          {
            v62[0] = 0;
            v62[1] = 0;
            v63 = v42;
            if ( v42 != 0 && v42 != -4096 && v42 != -8192 )
            {
              sub_BD73F0((__int64)v62);
              v42 = v63;
            }
            v44 = *(_DWORD *)(a1 + 56);
            if ( v44 )
            {
              v45 = v44 - 1;
              v46 = *(_QWORD *)(a1 + 40);
              v47 = v45 & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
              v48 = (_QWORD *)(v46 + 48LL * v47);
              v49 = v48[2];
              if ( v49 == v42 )
              {
LABEL_64:
                v64 = 0;
                v65 = 0;
                v66 = -8192;
                v50 = v48[2];
                if ( v50 != -8192 )
                {
                  if ( v50 && v50 != -4096 )
                  {
                    v60 = v48;
                    sub_BD60C0(v48);
                    v48 = v60;
                  }
                  v48[2] = -8192;
                  if ( v66 != 0 && v66 != -4096 && v66 != -8192 )
                    sub_BD60C0(&v64);
                  v42 = v63;
                }
                --*(_DWORD *)(a1 + 48);
                ++*(_DWORD *)(a1 + 52);
              }
              else
              {
                v53 = 1;
                while ( v49 != -4096 )
                {
                  v54 = v53 + 1;
                  v47 = v45 & (v53 + v47);
                  v48 = (_QWORD *)(v46 + 48LL * v47);
                  v49 = v48[2];
                  if ( v49 == v42 )
                    goto LABEL_64;
                  v53 = v54;
                }
              }
            }
            if ( v42 != 0 && v42 != -4096 && v42 != -8192 )
              sub_BD60C0(v62);
            v51 = v43 + 1;
            if ( v43 + 1 == v41 )
              break;
            while ( 1 )
            {
              v42 = *v51;
              v43 = v51;
              if ( *v51 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( v41 == ++v51 )
                goto LABEL_79;
            }
          }
          while ( v51 != v41 );
LABEL_79:
          v39 = *(unsigned __int8 *)(v37 + 36);
        }
LABEL_54:
        if ( (_BYTE)v39 )
          goto LABEL_55;
      }
      _libc_free(*(_QWORD *)(v37 + 16), v39);
    }
LABEL_55:
    *(_QWORD *)v37 = -8192;
    --*(_DWORD *)(a1 + 80);
    ++*(_DWORD *)(a1 + 84);
    goto LABEL_2;
  }
  v58 = 1;
  while ( v38 != (_BYTE *)-4096LL )
  {
    v36 = (v34 - 1) & (v58 + v36);
    v37 = v35 + 72LL * v36;
    v38 = *(_BYTE **)v37;
    if ( v12 == *(_BYTE **)v37 )
      goto LABEL_49;
    ++v58;
  }
LABEL_2:
  result = *(unsigned int *)(a1 + 120);
  v5 = *(_QWORD *)(a1 + 104);
  if ( (_DWORD)result )
  {
    v6 = (result - 1) & (a2 ^ (a2 >> 9));
    v61 = (__int64 *)(v5 + 80LL * v6);
    v7 = *v61;
    if ( a2 == *v61 )
    {
LABEL_4:
      result = v5 + 80 * result;
      if ( v61 != (__int64 *)result )
      {
        v8 = v61[2];
        v9 = v61[3];
        if ( v9 != v8 )
        {
          do
          {
            v11 = *(_QWORD *)(v8 + 8) & 7LL;
            if ( (unsigned int)v11 <= 2 )
            {
              v10 = *(_QWORD *)(v8 + 8) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v10 )
                sub_1029EF0(a1 + 128, v10, a2);
            }
            else if ( (_DWORD)v11 != 3 )
            {
              goto LABEL_115;
            }
            v8 += 16;
          }
          while ( v9 != v8 );
          v8 = v61[2];
        }
        if ( v8 )
          j_j___libc_free_0(v8, v61[4] - v8);
        result = (unsigned __int64)v61;
        *v61 = -16;
        --*(_DWORD *)(a1 + 112);
        ++*(_DWORD *)(a1 + 116);
      }
    }
    else
    {
      v52 = 1;
      while ( v7 != -4 )
      {
        v6 = (result - 1) & (v52 + v6);
        v61 = (__int64 *)(v5 + 80LL * v6);
        v7 = *v61;
        if ( a2 == *v61 )
          goto LABEL_4;
        ++v52;
      }
    }
  }
  return result;
}
