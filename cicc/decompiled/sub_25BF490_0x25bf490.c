// Function: sub_25BF490
// Address: 0x25bf490
//
__int64 __fastcall sub_25BF490(__int64 a1)
{
  __int64 v1; // r12
  __int64 v3; // rax
  __int64 result; // rax
  __int64 *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // rsi
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r9
  unsigned int v12; // edx
  int v13; // edx
  __int64 v14; // rax
  unsigned int v15; // r14d
  unsigned int v16; // esi
  __int64 v17; // rcx
  __int64 v18; // rdi
  __int64 *v19; // r9
  int v20; // r15d
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // r11
  int v24; // eax
  _BYTE *v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // r8
  _QWORD *v28; // r10
  int v29; // r15d
  unsigned int v30; // ecx
  _QWORD *v31; // rax
  __int64 v32; // rdx
  _DWORD *v33; // rax
  __int64 v34; // rax
  _QWORD *v35; // rdx
  _BYTE *v36; // r14
  unsigned int v37; // esi
  int v38; // eax
  __int64 v39; // rdi
  int v40; // esi
  __int64 v41; // r8
  int v42; // edx
  unsigned int v43; // ecx
  __int64 v44; // rax
  __int64 v45; // rax
  int v46; // eax
  int v47; // eax
  __int64 v48; // rdi
  int v49; // esi
  __int64 v50; // r8
  _QWORD *v51; // r9
  int v52; // r11d
  unsigned int v53; // ecx
  __int64 v54; // rax
  int v55; // r10d
  int v56; // eax
  int v57; // edx
  int v58; // eax
  int v59; // edx
  __int64 v60; // rsi
  unsigned int v61; // eax
  __int64 v62; // rdi
  int v63; // eax
  int v64; // r10d
  __int64 *v65; // r8
  int v66; // eax
  int v67; // r11d
  __int64 v68; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v69; // [rsp+8h] [rbp-38h] BYREF

  v1 = a1 + 64;
  v3 = *(_QWORD *)(a1 + 64);
  if ( *(_QWORD *)(a1 + 72) != v3 )
    *(_QWORD *)(a1 + 72) = v3;
  for ( result = *(_QWORD *)(a1 + 96); *(_QWORD *)(a1 + 88) != result; result = *(_QWORD *)(a1 + 96) )
  {
    while ( 1 )
    {
      v5 = *(__int64 **)(result - 16);
      if ( v5 == (__int64 *)(*(_QWORD *)(*(_QWORD *)(result - 24) + 16LL)
                           + 8LL * *(unsigned int *)(*(_QWORD *)(result - 24) + 24LL)) )
        break;
      *(_QWORD *)(result - 16) = v5 + 1;
      v6 = *(unsigned int *)(a1 + 32);
      v7 = *v5;
      v8 = *(_QWORD *)(a1 + 16);
      if ( !(_DWORD)v6 )
        goto LABEL_12;
      v9 = (v6 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( v7 == *v10 )
      {
LABEL_7:
        if ( v10 == (__int64 *)(v8 + 16 * v6) )
          goto LABEL_12;
        result = *(_QWORD *)(a1 + 96);
        v12 = *((_DWORD *)v10 + 2);
        if ( v12 < *(_DWORD *)(result - 8) )
        {
          *(_DWORD *)(result - 8) = v12;
          result = *(_QWORD *)(a1 + 96);
        }
      }
      else
      {
        v13 = 1;
        while ( v11 != -4096 )
        {
          v55 = v13 + 1;
          v9 = (v6 - 1) & (v13 + v9);
          v10 = (__int64 *)(v8 + 16LL * v9);
          v11 = *v10;
          if ( v7 == *v10 )
            goto LABEL_7;
          v13 = v55;
        }
LABEL_12:
        sub_25BE460((int *)a1, v7);
        result = *(_QWORD *)(a1 + 96);
      }
    }
    v68 = *(_QWORD *)(result - 24);
    v14 = result - 24;
    v15 = *(_DWORD *)(v14 + 16);
    *(_QWORD *)(a1 + 96) = v14;
    if ( *(_QWORD *)(a1 + 88) != v14 && *(_DWORD *)(v14 - 8) > v15 )
      *(_DWORD *)(v14 - 8) = v15;
    v16 = *(_DWORD *)(a1 + 32);
    if ( v16 )
    {
      v17 = v68;
      v18 = *(_QWORD *)(a1 + 16);
      v19 = 0;
      v20 = 1;
      v21 = (v16 - 1) & (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4));
      v22 = (__int64 *)(v18 + 16LL * v21);
      v23 = *v22;
      if ( v68 == *v22 )
      {
LABEL_18:
        v24 = *((_DWORD *)v22 + 2);
        goto LABEL_19;
      }
      while ( v23 != -4096 )
      {
        if ( !v19 && v23 == -8192 )
          v19 = v22;
        v21 = (v16 - 1) & (v20 + v21);
        v22 = (__int64 *)(v18 + 16LL * v21);
        v23 = *v22;
        if ( v68 == *v22 )
          goto LABEL_18;
        ++v20;
      }
      if ( !v19 )
        v19 = v22;
      v56 = *(_DWORD *)(a1 + 24);
      ++*(_QWORD *)(a1 + 8);
      v57 = v56 + 1;
      v69 = v19;
      if ( 4 * (v56 + 1) < 3 * v16 )
      {
        if ( v16 - *(_DWORD *)(a1 + 28) - v57 <= v16 >> 3 )
        {
          sub_25BE280(a1 + 8, v16);
          v58 = *(_DWORD *)(a1 + 32);
          if ( v58 )
          {
            v17 = v68;
            v59 = v58 - 1;
            v60 = *(_QWORD *)(a1 + 16);
            v61 = (v58 - 1) & (((unsigned int)v68 >> 9) ^ ((unsigned int)v68 >> 4));
            v19 = (__int64 *)(v60 + 16LL * v61);
            v62 = *v19;
            if ( v68 == *v19 )
            {
LABEL_73:
              v63 = *(_DWORD *)(a1 + 24);
              v69 = v19;
              v57 = v63 + 1;
            }
            else
            {
              v64 = 1;
              v65 = 0;
              while ( v62 != -4096 )
              {
                if ( !v65 && v62 == -8192 )
                  v65 = v19;
                v61 = v59 & (v64 + v61);
                v19 = (__int64 *)(v60 + 16LL * v61);
                v62 = *v19;
                if ( v68 == *v19 )
                  goto LABEL_73;
                ++v64;
              }
              if ( !v65 )
                v65 = v19;
              v57 = *(_DWORD *)(a1 + 24) + 1;
              v69 = v65;
              v19 = v65;
            }
          }
          else
          {
            v66 = *(_DWORD *)(a1 + 24);
            v69 = 0;
            v19 = 0;
            v17 = v68;
            v57 = v66 + 1;
          }
        }
        goto LABEL_66;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 8);
      v69 = 0;
    }
    sub_25BE280(a1 + 8, 2 * v16);
    sub_25BC7E0(a1 + 8, &v68, &v69);
    v17 = v68;
    v19 = v69;
    v57 = *(_DWORD *)(a1 + 24) + 1;
LABEL_66:
    *(_DWORD *)(a1 + 24) = v57;
    if ( *v19 != -4096 )
      --*(_DWORD *)(a1 + 28);
    *v19 = v17;
    v24 = 0;
    *((_DWORD *)v19 + 2) = 0;
LABEL_19:
    if ( v15 == v24 )
    {
      v25 = *(_BYTE **)(a1 + 72);
      while ( 1 )
      {
        v34 = *(_QWORD *)(a1 + 48);
        v35 = (_QWORD *)(v34 - 8);
        if ( v25 == *(_BYTE **)(a1 + 80) )
        {
          sub_25BCF70(v1, v25, v35);
          v36 = *(_BYTE **)(a1 + 72);
          v35 = (_QWORD *)(*(_QWORD *)(a1 + 48) - 8LL);
        }
        else
        {
          if ( v25 )
          {
            *(_QWORD *)v25 = *(_QWORD *)(v34 - 8);
            v25 = *(_BYTE **)(a1 + 72);
            v35 = (_QWORD *)(*(_QWORD *)(a1 + 48) - 8LL);
          }
          v36 = v25 + 8;
          *(_QWORD *)(a1 + 72) = v25 + 8;
        }
        v37 = *(_DWORD *)(a1 + 32);
        *(_QWORD *)(a1 + 48) = v35;
        if ( !v37 )
          break;
        v26 = *((_QWORD *)v36 - 1);
        v27 = *(_QWORD *)(a1 + 16);
        v28 = 0;
        v29 = 1;
        v30 = (v37 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v31 = (_QWORD *)(v27 + 16LL * v30);
        v32 = *v31;
        if ( *v31 != v26 )
        {
          while ( v32 != -4096 )
          {
            if ( v32 == -8192 && !v28 )
              v28 = v31;
            v30 = (v37 - 1) & (v29 + v30);
            v31 = (_QWORD *)(v27 + 16LL * v30);
            v32 = *v31;
            if ( v26 == *v31 )
              goto LABEL_24;
            ++v29;
          }
          if ( !v28 )
            v28 = v31;
          v46 = *(_DWORD *)(a1 + 24);
          ++*(_QWORD *)(a1 + 8);
          v42 = v46 + 1;
          if ( 4 * (v46 + 1) < 3 * v37 )
          {
            if ( v37 - *(_DWORD *)(a1 + 28) - v42 <= v37 >> 3 )
            {
              sub_25BE280(a1 + 8, v37);
              v47 = *(_DWORD *)(a1 + 32);
              if ( !v47 )
              {
LABEL_97:
                ++*(_DWORD *)(a1 + 24);
                BUG();
              }
              v48 = *((_QWORD *)v36 - 1);
              v49 = v47 - 1;
              v50 = *(_QWORD *)(a1 + 16);
              v51 = 0;
              v52 = 1;
              v42 = *(_DWORD *)(a1 + 24) + 1;
              v53 = (v47 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
              v28 = (_QWORD *)(v50 + 16LL * v53);
              v54 = *v28;
              if ( *v28 != v48 )
              {
                while ( v54 != -4096 )
                {
                  if ( !v51 && v54 == -8192 )
                    v51 = v28;
                  v53 = v49 & (v52 + v53);
                  v28 = (_QWORD *)(v50 + 16LL * v53);
                  v54 = *v28;
                  if ( v48 == *v28 )
                    goto LABEL_34;
                  ++v52;
                }
LABEL_50:
                if ( v51 )
                  v28 = v51;
              }
            }
LABEL_34:
            *(_DWORD *)(a1 + 24) = v42;
            if ( *v28 != -4096 )
              --*(_DWORD *)(a1 + 28);
            v45 = *((_QWORD *)v36 - 1);
            *((_DWORD *)v28 + 2) = 0;
            *v28 = v45;
            v33 = v28 + 1;
            goto LABEL_25;
          }
LABEL_32:
          sub_25BE280(a1 + 8, 2 * v37);
          v38 = *(_DWORD *)(a1 + 32);
          if ( !v38 )
            goto LABEL_97;
          v39 = *((_QWORD *)v36 - 1);
          v40 = v38 - 1;
          v41 = *(_QWORD *)(a1 + 16);
          v42 = *(_DWORD *)(a1 + 24) + 1;
          v43 = (v38 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
          v28 = (_QWORD *)(v41 + 16LL * v43);
          v44 = *v28;
          if ( *v28 != v39 )
          {
            v67 = 1;
            v51 = 0;
            while ( v44 != -4096 )
            {
              if ( v44 == -8192 && !v51 )
                v51 = v28;
              v43 = v40 & (v67 + v43);
              v28 = (_QWORD *)(v41 + 16LL * v43);
              v44 = *v28;
              if ( v39 == *v28 )
                goto LABEL_34;
              ++v67;
            }
            goto LABEL_50;
          }
          goto LABEL_34;
        }
LABEL_24:
        v33 = v31 + 1;
LABEL_25:
        *v33 = -1;
        v25 = *(_BYTE **)(a1 + 72);
        result = v68;
        if ( *((_QWORD *)v25 - 1) == v68 )
          return result;
      }
      ++*(_QWORD *)(a1 + 8);
      goto LABEL_32;
    }
  }
  return result;
}
