// Function: sub_26D5040
// Address: 0x26d5040
//
__int64 __fastcall sub_26D5040(__int64 a1)
{
  __int64 v1; // r13
  __int64 v3; // rax
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  unsigned int v8; // r12d
  unsigned int v9; // esi
  __int64 v10; // rcx
  __int64 v11; // rdi
  __int64 *v12; // r9
  int v13; // r15d
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r11
  int v17; // eax
  _BYTE *v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // r8
  _QWORD *v21; // r10
  int v22; // r15d
  unsigned int v23; // ecx
  _QWORD *v24; // rax
  __int64 v25; // rdx
  _DWORD *v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // rdx
  _BYTE *v29; // r12
  unsigned int v30; // esi
  int v31; // eax
  __int64 v32; // rsi
  int v33; // r8d
  __int64 v34; // rdi
  int v35; // edx
  unsigned int v36; // ecx
  __int64 v37; // rax
  __int64 v38; // rax
  int v39; // eax
  int v40; // eax
  __int64 v41; // rsi
  int v42; // eax
  __int64 v43; // rdi
  _QWORD *v44; // r9
  int v45; // r11d
  unsigned int v46; // ecx
  __int64 v47; // r8
  int v48; // eax
  int v49; // edx
  int v50; // r11d
  __int64 v51; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v52; // [rsp+8h] [rbp-38h] BYREF

  v1 = a1 + 64;
  v3 = *(_QWORD *)(a1 + 64);
  if ( *(_QWORD *)(a1 + 72) != v3 )
    *(_QWORD *)(a1 + 72) = v3;
  while ( 1 )
  {
    result = *(_QWORD *)(a1 + 96);
    if ( *(_QWORD *)(a1 + 88) == result )
      return result;
    sub_26D4F40(a1);
    v5 = *(_QWORD *)(a1 + 96);
    v6 = *(_QWORD *)(v5 - 24);
    v7 = v5 - 24;
    v51 = v6;
    v8 = *(_DWORD *)(v7 + 16);
    *(_QWORD *)(a1 + 96) = v7;
    if ( *(_QWORD *)(a1 + 88) != v7 && *(_DWORD *)(v7 - 8) > v8 )
      *(_DWORD *)(v7 - 8) = v8;
    v9 = *(_DWORD *)(a1 + 32);
    if ( !v9 )
    {
      ++*(_QWORD *)(a1 + 8);
      v52 = 0;
      goto LABEL_58;
    }
    v10 = v51;
    v11 = *(_QWORD *)(a1 + 16);
    v12 = 0;
    v13 = 1;
    v14 = (v9 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
    v15 = (__int64 *)(v11 + 16LL * v14);
    v16 = *v15;
    if ( v51 != *v15 )
    {
      while ( v16 != -4096 )
      {
        if ( !v12 && v16 == -8192 )
          v12 = v15;
        v14 = (v9 - 1) & (v13 + v14);
        v15 = (__int64 *)(v11 + 16LL * v14);
        v16 = *v15;
        if ( v51 == *v15 )
          goto LABEL_9;
        ++v13;
      }
      if ( !v12 )
        v12 = v15;
      v48 = *(_DWORD *)(a1 + 24);
      ++*(_QWORD *)(a1 + 8);
      v49 = v48 + 1;
      v52 = v12;
      if ( 4 * (v48 + 1) >= 3 * v9 )
      {
LABEL_58:
        v9 *= 2;
      }
      else if ( v9 - *(_DWORD *)(a1 + 28) - v49 > v9 >> 3 )
      {
LABEL_54:
        *(_DWORD *)(a1 + 24) = v49;
        if ( *v12 != -4096 )
          --*(_DWORD *)(a1 + 28);
        *v12 = v10;
        v17 = 0;
        *((_DWORD *)v12 + 2) = 0;
        goto LABEL_10;
      }
      sub_26D4B60(a1 + 8, v9);
      sub_26C98D0(a1 + 8, &v51, &v52);
      v10 = v51;
      v12 = v52;
      v49 = *(_DWORD *)(a1 + 24) + 1;
      goto LABEL_54;
    }
LABEL_9:
    v17 = *((_DWORD *)v15 + 2);
LABEL_10:
    if ( v8 == v17 )
    {
      v18 = *(_BYTE **)(a1 + 72);
      while ( 1 )
      {
        v27 = *(_QWORD *)(a1 + 48);
        v28 = (_QWORD *)(v27 - 8);
        if ( *(_BYTE **)(a1 + 80) == v18 )
        {
          sub_26C7040(v1, v18, v28);
          v29 = *(_BYTE **)(a1 + 72);
          v28 = (_QWORD *)(*(_QWORD *)(a1 + 48) - 8LL);
        }
        else
        {
          if ( v18 )
          {
            *(_QWORD *)v18 = *(_QWORD *)(v27 - 8);
            v18 = *(_BYTE **)(a1 + 72);
            v28 = (_QWORD *)(*(_QWORD *)(a1 + 48) - 8LL);
          }
          v29 = v18 + 8;
          *(_QWORD *)(a1 + 72) = v18 + 8;
        }
        v30 = *(_DWORD *)(a1 + 32);
        *(_QWORD *)(a1 + 48) = v28;
        if ( !v30 )
          break;
        v19 = *((_QWORD *)v29 - 1);
        v20 = *(_QWORD *)(a1 + 16);
        v21 = 0;
        v22 = 1;
        v23 = (v30 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v24 = (_QWORD *)(v20 + 16LL * v23);
        v25 = *v24;
        if ( v19 != *v24 )
        {
          while ( v25 != -4096 )
          {
            if ( !v21 && v25 == -8192 )
              v21 = v24;
            v23 = (v30 - 1) & (v22 + v23);
            v24 = (_QWORD *)(v20 + 16LL * v23);
            v25 = *v24;
            if ( v19 == *v24 )
              goto LABEL_13;
            ++v22;
          }
          if ( !v21 )
            v21 = v24;
          v39 = *(_DWORD *)(a1 + 24);
          ++*(_QWORD *)(a1 + 8);
          v35 = v39 + 1;
          if ( 4 * (v39 + 1) < 3 * v30 )
          {
            if ( v30 - *(_DWORD *)(a1 + 28) - v35 <= v30 >> 3 )
            {
              sub_26D4B60(a1 + 8, v30);
              v40 = *(_DWORD *)(a1 + 32);
              if ( !v40 )
              {
LABEL_72:
                ++*(_DWORD *)(a1 + 24);
                BUG();
              }
              v41 = *((_QWORD *)v29 - 1);
              v42 = v40 - 1;
              v43 = *(_QWORD *)(a1 + 16);
              v44 = 0;
              v45 = 1;
              v46 = v42 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
              v35 = *(_DWORD *)(a1 + 24) + 1;
              v21 = (_QWORD *)(v43 + 16LL * v46);
              v47 = *v21;
              if ( *v21 != v41 )
              {
                while ( v47 != -4096 )
                {
                  if ( !v44 && v47 == -8192 )
                    v44 = v21;
                  v46 = v42 & (v45 + v46);
                  v21 = (_QWORD *)(v43 + 16LL * v46);
                  v47 = *v21;
                  if ( v41 == *v21 )
                    goto LABEL_23;
                  ++v45;
                }
LABEL_39:
                if ( v44 )
                  v21 = v44;
              }
            }
LABEL_23:
            *(_DWORD *)(a1 + 24) = v35;
            if ( *v21 != -4096 )
              --*(_DWORD *)(a1 + 28);
            v38 = *((_QWORD *)v29 - 1);
            *((_DWORD *)v21 + 2) = 0;
            *v21 = v38;
            v26 = v21 + 1;
            goto LABEL_14;
          }
LABEL_21:
          sub_26D4B60(a1 + 8, 2 * v30);
          v31 = *(_DWORD *)(a1 + 32);
          if ( !v31 )
            goto LABEL_72;
          v32 = *((_QWORD *)v29 - 1);
          v33 = v31 - 1;
          v34 = *(_QWORD *)(a1 + 16);
          v35 = *(_DWORD *)(a1 + 24) + 1;
          v36 = (v31 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
          v21 = (_QWORD *)(v34 + 16LL * v36);
          v37 = *v21;
          if ( *v21 != v32 )
          {
            v50 = 1;
            v44 = 0;
            while ( v37 != -4096 )
            {
              if ( !v44 && v37 == -8192 )
                v44 = v21;
              v36 = v33 & (v50 + v36);
              v21 = (_QWORD *)(v34 + 16LL * v36);
              v37 = *v21;
              if ( v32 == *v21 )
                goto LABEL_23;
              ++v50;
            }
            goto LABEL_39;
          }
          goto LABEL_23;
        }
LABEL_13:
        v26 = v24 + 1;
LABEL_14:
        *v26 = -1;
        v18 = *(_BYTE **)(a1 + 72);
        result = v51;
        if ( *((_QWORD *)v18 - 1) == v51 )
          return result;
      }
      ++*(_QWORD *)(a1 + 8);
      goto LABEL_21;
    }
  }
}
