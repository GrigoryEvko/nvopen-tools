// Function: sub_3505E20
// Address: 0x3505e20
//
__int64 __fastcall sub_3505E20(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // r14
  __int64 v9; // r13
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r12
  unsigned __int8 v13; // al
  __int64 v14; // r9
  unsigned __int8 **v15; // rcx
  unsigned int v16; // esi
  int v17; // r11d
  unsigned int v18; // ecx
  _QWORD *v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 *v23; // rax
  __int64 v24; // rax
  __int64 *v25; // rax
  unsigned __int8 v26; // al
  __int64 v27; // rdx
  unsigned __int8 **v28; // rcx
  __int64 v29; // rbx
  unsigned int v30; // esi
  __int64 v31; // r10
  unsigned int v32; // r11d
  __int64 *v33; // rdi
  unsigned int v34; // ecx
  __int64 *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rdx
  int v38; // ecx
  int v39; // eax
  __int64 v40; // r11
  unsigned int v41; // edx
  __int64 v42; // rdi
  int v43; // r10d
  _QWORD *v44; // rsi
  int v45; // eax
  __int64 v46; // r11
  int v47; // r10d
  unsigned int v48; // edx
  __int64 v49; // rdi
  int v50; // r11d
  int v51; // r11d
  __int64 v52; // r10
  unsigned int v53; // ecx
  int v54; // edx
  int v55; // edx
  int v56; // r10d
  int v57; // r10d
  __int64 *v58; // rcx
  unsigned int v59; // r12d
  int v60; // esi
  __int64 v61; // rdi
  int v62; // edi
  __int64 *v63; // rsi
  unsigned int v64; // [rsp+4h] [rbp-6Ch]
  __int64 v65; // [rsp+8h] [rbp-68h]
  __int64 v67; // [rsp+20h] [rbp-50h]
  __int64 v69; // [rsp+30h] [rbp-40h]
  __int64 v70; // [rsp+38h] [rbp-38h]

  result = *a1 + 320LL;
  v65 = result;
  v67 = *(_QWORD *)(*a1 + 328LL);
  if ( v67 == result )
    return result;
  do
  {
    v8 = 0;
    v9 = 0;
    v69 = 0;
    v10 = *(_QWORD *)(v67 + 56);
    v70 = v67 + 48;
    if ( v67 + 48 == v10 )
      goto LABEL_33;
    do
    {
      while ( 1 )
      {
        if ( (*(_BYTE *)(*(_QWORD *)(v10 + 16) + 24LL) & 0x10) != 0 )
          goto LABEL_20;
        v11 = sub_B10CD0(v10 + 56);
        v12 = v11;
        if ( !v11 || v11 == v9 )
        {
          v69 = v10;
          goto LABEL_20;
        }
        if ( !v8 )
        {
          v69 = v10;
          v8 = v10;
          v9 = v11;
          goto LABEL_20;
        }
        if ( v9 )
        {
          v13 = *(_BYTE *)(v9 - 16);
          if ( (v13 & 2) != 0 )
          {
            if ( *(_DWORD *)(v9 - 24) == 2 )
              v14 = *(_QWORD *)(*(_QWORD *)(v9 - 32) + 8LL);
            else
              v14 = 0;
            v15 = *(unsigned __int8 ***)(v9 - 32);
          }
          else
          {
            v37 = v9 - 16;
            v14 = 0;
            if ( ((*(_WORD *)(v9 - 16) >> 6) & 0xF) == 2 )
              v14 = *(_QWORD *)(v37 - 8LL * ((v13 >> 2) & 0xF) + 8);
            v15 = (unsigned __int8 **)(v37 - 8LL * ((v13 >> 2) & 0xF));
          }
          v9 = sub_35057B0(a1, *v15, v14);
        }
        v16 = *(_DWORD *)(a3 + 24);
        if ( !v16 )
        {
          ++*(_QWORD *)a3;
          goto LABEL_58;
        }
        a6 = *(_QWORD *)(a3 + 8);
        v17 = 1;
        a5 = 0;
        v18 = (v16 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v19 = (_QWORD *)(a6 + 16LL * v18);
        v20 = *v19;
        if ( v8 != *v19 )
        {
          while ( v20 != -4096 )
          {
            if ( !a5 && v20 == -8192 )
              a5 = (__int64)v19;
            v18 = (v16 - 1) & (v17 + v18);
            v19 = (_QWORD *)(a6 + 16LL * v18);
            v20 = *v19;
            if ( v8 == *v19 )
              goto LABEL_17;
            ++v17;
          }
          if ( a5 )
            v19 = (_QWORD *)a5;
          ++*(_QWORD *)a3;
          v38 = *(_DWORD *)(a3 + 16) + 1;
          if ( 4 * v38 >= 3 * v16 )
          {
LABEL_58:
            sub_3505C40(a3, 2 * v16);
            v39 = *(_DWORD *)(a3 + 24);
            if ( !v39 )
              goto LABEL_120;
            a5 = (unsigned int)(v39 - 1);
            v40 = *(_QWORD *)(a3 + 8);
            v41 = a5 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
            v38 = *(_DWORD *)(a3 + 16) + 1;
            v19 = (_QWORD *)(v40 + 16LL * v41);
            v42 = *v19;
            if ( v8 != *v19 )
            {
              v43 = 1;
              v44 = 0;
              while ( v42 != -4096 )
              {
                if ( !v44 && v42 == -8192 )
                  v44 = v19;
                a6 = (unsigned int)(v43 + 1);
                v41 = a5 & (v43 + v41);
                v19 = (_QWORD *)(v40 + 16LL * v41);
                v42 = *v19;
                if ( v8 == *v19 )
                  goto LABEL_54;
                ++v43;
              }
LABEL_73:
              if ( v44 )
                v19 = v44;
            }
          }
          else
          {
            a6 = v16 >> 3;
            if ( v16 - *(_DWORD *)(a3 + 20) - v38 <= (unsigned int)a6 )
            {
              v64 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
              sub_3505C40(a3, v16);
              v45 = *(_DWORD *)(a3 + 24);
              if ( !v45 )
              {
LABEL_120:
                ++*(_DWORD *)(a3 + 16);
                BUG();
              }
              a5 = (unsigned int)(v45 - 1);
              v46 = *(_QWORD *)(a3 + 8);
              v47 = 1;
              v48 = a5 & v64;
              v38 = *(_DWORD *)(a3 + 16) + 1;
              v44 = 0;
              v19 = (_QWORD *)(v46 + 16LL * ((unsigned int)a5 & v64));
              v49 = *v19;
              if ( v8 != *v19 )
              {
                while ( v49 != -4096 )
                {
                  if ( v49 == -8192 && !v44 )
                    v44 = v19;
                  a6 = (unsigned int)(v47 + 1);
                  v48 = a5 & (v47 + v48);
                  v19 = (_QWORD *)(v46 + 16LL * v48);
                  v49 = *v19;
                  if ( v8 == *v19 )
                    goto LABEL_54;
                  ++v47;
                }
                goto LABEL_73;
              }
            }
          }
LABEL_54:
          *(_DWORD *)(a3 + 16) = v38;
          if ( *v19 != -4096 )
            --*(_DWORD *)(a3 + 20);
          *v19 = v8;
          v19[1] = 0;
        }
LABEL_17:
        v19[1] = v9;
        v21 = *(unsigned int *)(a2 + 8);
        if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          sub_C8D5F0(a2, (const void *)(a2 + 16), v21 + 1, 0x10u, a5, a6);
          v21 = *(unsigned int *)(a2 + 8);
        }
        v22 = v69;
        v23 = (__int64 *)(*(_QWORD *)a2 + 16 * v21);
        v69 = v10;
        *v23 = v8;
        v9 = v12;
        v8 = v10;
        v23[1] = v22;
        ++*(_DWORD *)(a2 + 8);
LABEL_20:
        if ( (*(_BYTE *)v10 & 4) == 0 )
          break;
        v10 = *(_QWORD *)(v10 + 8);
        if ( v70 == v10 )
          goto LABEL_22;
      }
      while ( (*(_BYTE *)(v10 + 44) & 8) != 0 )
        v10 = *(_QWORD *)(v10 + 8);
      v10 = *(_QWORD *)(v10 + 8);
    }
    while ( v70 != v10 );
LABEL_22:
    if ( v69 == 0 || v8 == 0 || !v9 )
      goto LABEL_33;
    v24 = *(unsigned int *)(a2 + 8);
    if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), v24 + 1, 0x10u, a5, a6);
      v24 = *(unsigned int *)(a2 + 8);
    }
    v25 = (__int64 *)(*(_QWORD *)a2 + 16 * v24);
    *v25 = v8;
    v25[1] = v69;
    ++*(_DWORD *)(a2 + 8);
    v26 = *(_BYTE *)(v9 - 16);
    if ( (v26 & 2) != 0 )
    {
      if ( *(_DWORD *)(v9 - 24) == 2 )
        v27 = *(_QWORD *)(*(_QWORD *)(v9 - 32) + 8LL);
      else
        v27 = 0;
      v28 = *(unsigned __int8 ***)(v9 - 32);
    }
    else
    {
      if ( ((*(_WORD *)(v9 - 16) >> 6) & 0xF) == 2 )
        v27 = *(_QWORD *)(v9 - 16 - 8LL * ((v26 >> 2) & 0xF) + 8);
      else
        v27 = 0;
      v28 = (unsigned __int8 **)(v9 - 16 - 8LL * ((v26 >> 2) & 0xF));
    }
    v29 = sub_35057B0(a1, *v28, v27);
    v30 = *(_DWORD *)(a3 + 24);
    if ( !v30 )
    {
      ++*(_QWORD *)a3;
LABEL_78:
      sub_3505C40(a3, 2 * v30);
      v50 = *(_DWORD *)(a3 + 24);
      if ( v50 )
      {
        v51 = v50 - 1;
        v52 = *(_QWORD *)(a3 + 8);
        v53 = v51 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v54 = *(_DWORD *)(a3 + 16) + 1;
        v35 = (__int64 *)(v52 + 16LL * v53);
        a5 = *v35;
        if ( v8 != *v35 )
        {
          v62 = 1;
          v63 = 0;
          while ( a5 != -4096 )
          {
            if ( !v63 && a5 == -8192 )
              v63 = v35;
            a6 = (unsigned int)(v62 + 1);
            v53 = v51 & (v62 + v53);
            v35 = (__int64 *)(v52 + 16LL * v53);
            a5 = *v35;
            if ( *v35 == v8 )
              goto LABEL_80;
            ++v62;
          }
          if ( v63 )
            v35 = v63;
        }
        goto LABEL_80;
      }
      goto LABEL_121;
    }
    v31 = *(_QWORD *)(a3 + 8);
    v32 = v30 - 1;
    a5 = 1;
    v33 = 0;
    v34 = (v30 - 1) & (((unsigned int)v8 >> 4) ^ ((unsigned int)v8 >> 9));
    v35 = (__int64 *)(v31 + 16LL * v34);
    v36 = *v35;
    if ( *v35 == v8 )
      goto LABEL_32;
    while ( v36 != -4096 )
    {
      if ( !v33 && v36 == -8192 )
        v33 = v35;
      a6 = (unsigned int)(a5 + 1);
      a5 = v34 + (unsigned int)a5;
      v34 = v32 & a5;
      v35 = (__int64 *)(v31 + 16LL * (v32 & (unsigned int)a5));
      v36 = *v35;
      if ( *v35 == v8 )
        goto LABEL_32;
      a5 = (unsigned int)a6;
    }
    if ( v33 )
      v35 = v33;
    v55 = *(_DWORD *)(a3 + 16);
    ++*(_QWORD *)a3;
    v54 = v55 + 1;
    if ( 4 * v54 >= 3 * v30 )
      goto LABEL_78;
    if ( v30 - *(_DWORD *)(a3 + 20) - v54 <= v30 >> 3 )
    {
      sub_3505C40(a3, v30);
      v56 = *(_DWORD *)(a3 + 24);
      if ( v56 )
      {
        v57 = v56 - 1;
        a6 = *(_QWORD *)(a3 + 8);
        v58 = 0;
        v59 = v57 & (((unsigned int)v8 >> 4) ^ ((unsigned int)v8 >> 9));
        v60 = 1;
        v54 = *(_DWORD *)(a3 + 16) + 1;
        v35 = (__int64 *)(a6 + 16LL * v59);
        v61 = *v35;
        if ( *v35 != v8 )
        {
          while ( v61 != -4096 )
          {
            if ( v61 == -8192 && !v58 )
              v58 = v35;
            a5 = (unsigned int)(v60 + 1);
            v59 = v57 & (v60 + v59);
            v35 = (__int64 *)(a6 + 16LL * v59);
            v61 = *v35;
            if ( *v35 == v8 )
              goto LABEL_80;
            ++v60;
          }
          if ( v58 )
            v35 = v58;
        }
        goto LABEL_80;
      }
LABEL_121:
      ++*(_DWORD *)(a3 + 16);
      BUG();
    }
LABEL_80:
    *(_DWORD *)(a3 + 16) = v54;
    if ( *v35 != -4096 )
      --*(_DWORD *)(a3 + 20);
    *v35 = v8;
    v35[1] = 0;
LABEL_32:
    v35[1] = v29;
LABEL_33:
    result = *(_QWORD *)(v67 + 8);
    v67 = result;
  }
  while ( v65 != result );
  return result;
}
