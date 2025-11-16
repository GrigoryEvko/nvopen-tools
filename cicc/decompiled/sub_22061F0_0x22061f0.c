// Function: sub_22061F0
// Address: 0x22061f0
//
__int64 __fastcall sub_22061F0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rax
  int v7; // eax
  unsigned __int16 v8; // ax
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  unsigned int v13; // esi
  __int64 v14; // r10
  __int64 v15; // r8
  __int64 v16; // rdx
  _QWORD *v17; // rcx
  __int64 v18; // rdi
  int v19; // r14d
  _QWORD *v20; // rax
  int v21; // edx
  unsigned int v22; // esi
  __int64 v23; // r9
  __int64 v24; // r10
  unsigned int v25; // r14d
  __int64 v26; // r8
  __int64 v27; // rdi
  int v28; // ecx
  _QWORD *v29; // rdx
  int v30; // r8d
  int v31; // r8d
  __int64 v32; // r10
  _QWORD *v33; // rdi
  unsigned int v34; // r9d
  int v35; // ecx
  __int64 v36; // rsi
  int v37; // r8d
  int v38; // r8d
  __int64 v39; // r9
  unsigned int v40; // ecx
  __int64 v41; // rdi
  int v42; // esi
  _QWORD *v43; // r10
  int v44; // r8d
  int v45; // r8d
  __int64 v46; // r9
  unsigned int v47; // r15d
  int v48; // ecx
  __int64 v49; // rsi
  int v50; // r9d
  int v51; // r9d
  __int64 v52; // r10
  unsigned int v53; // ecx
  __int64 v54; // r8
  int v55; // edi
  _QWORD *v56; // rsi
  __int64 v57; // [rsp+8h] [rbp-88h]
  __int64 v59; // [rsp+18h] [rbp-78h]
  int v60; // [rsp+2Ch] [rbp-64h] BYREF
  _BYTE v61[96]; // [rsp+30h] [rbp-60h] BYREF

  result = a2 + 320;
  v57 = a2 + 320;
  v59 = *(_QWORD *)(a2 + 328);
  if ( v59 != a2 + 320 )
  {
    v3 = a1 + 312;
    while ( 1 )
    {
      v4 = *(_QWORD *)(v59 + 32);
      v5 = v59 + 24;
      if ( v4 != v59 + 24 )
        break;
LABEL_26:
      result = *(_QWORD *)(v59 + 8);
      v59 = result;
      if ( v57 == result )
        return result;
    }
    while ( 1 )
    {
      v8 = **(_WORD **)(v4 + 16);
      if ( v8 > 0x5BFu )
        break;
      if ( v8 > 0x5BAu )
        goto LABEL_38;
      if ( v8 == 324 )
      {
        v60 = *(_DWORD *)(*(_QWORD *)(v4 + 32) + 8LL);
        sub_2205FC0((__int64)v61, v3, &v60);
        v13 = *(_DWORD *)(a1 + 304);
        v14 = a1 + 280;
        if ( !v13 )
        {
          ++*(_QWORD *)(a1 + 280);
          goto LABEL_77;
        }
        v15 = *(_QWORD *)(a1 + 288);
        LODWORD(v16) = (v13 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v17 = (_QWORD *)(v15 + 8LL * (unsigned int)v16);
        v18 = *v17;
        if ( v4 != *v17 )
        {
          v19 = 1;
          v20 = 0;
          while ( v18 != -8 )
          {
            if ( !v20 && v18 == -16 )
              v20 = v17;
            v16 = (v13 - 1) & ((_DWORD)v16 + v19);
            v17 = (_QWORD *)(v15 + 8 * v16);
            v18 = *v17;
            if ( v4 == *v17 )
              goto LABEL_14;
            ++v19;
          }
          if ( !v20 )
            v20 = v17;
          ++*(_QWORD *)(a1 + 280);
          v21 = *(_DWORD *)(a1 + 296) + 1;
          if ( 4 * v21 < 3 * v13 )
          {
            if ( v13 - *(_DWORD *)(a1 + 300) - v21 <= v13 >> 3 )
            {
              sub_1E22DE0(v14, v13);
              v44 = *(_DWORD *)(a1 + 304);
              if ( !v44 )
              {
LABEL_126:
                ++*(_DWORD *)(a1 + 296);
                BUG();
              }
              v45 = v44 - 1;
              v46 = *(_QWORD *)(a1 + 288);
              v33 = 0;
              v47 = v45 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
              v21 = *(_DWORD *)(a1 + 296) + 1;
              v48 = 1;
              v20 = (_QWORD *)(v46 + 8LL * v47);
              v49 = *v20;
              if ( v4 != *v20 )
              {
                while ( v49 != -8 )
                {
                  if ( v49 == -16 && !v33 )
                    v33 = v20;
                  v47 = v45 & (v48 + v47);
                  v20 = (_QWORD *)(v46 + 8LL * v47);
                  v49 = *v20;
                  if ( v4 == *v20 )
                    goto LABEL_59;
                  ++v48;
                }
LABEL_73:
                if ( v33 )
                  v20 = v33;
              }
            }
LABEL_59:
            *(_DWORD *)(a1 + 296) = v21;
            if ( *v20 != -8 )
              --*(_DWORD *)(a1 + 300);
            *v20 = v4;
            goto LABEL_14;
          }
LABEL_77:
          sub_1E22DE0(v14, 2 * v13);
          v37 = *(_DWORD *)(a1 + 304);
          if ( !v37 )
            goto LABEL_126;
          v38 = v37 - 1;
          v39 = *(_QWORD *)(a1 + 288);
          v40 = v38 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v21 = *(_DWORD *)(a1 + 296) + 1;
          v20 = (_QWORD *)(v39 + 8LL * v40);
          v41 = *v20;
          if ( v4 != *v20 )
          {
            v42 = 1;
            v43 = 0;
            while ( v41 != -8 )
            {
              if ( v41 == -16 && !v43 )
                v43 = v20;
              v40 = v38 & (v42 + v40);
              v20 = (_QWORD *)(v39 + 8LL * v40);
              v41 = *v20;
              if ( v4 == *v20 )
                goto LABEL_59;
              ++v42;
            }
            if ( v43 )
              v20 = v43;
          }
          goto LABEL_59;
        }
      }
      else
      {
        if ( v8 > 0x144u )
        {
          if ( v8 <= 0x56Fu )
          {
            if ( v8 <= 0x56Au && (unsigned __int16)(v8 - 325) > 1u )
              goto LABEL_14;
            goto LABEL_11;
          }
          if ( (unsigned __int16)(v8 - 1437) > 4u )
            goto LABEL_14;
          goto LABEL_29;
        }
        if ( v8 != 144 )
        {
          if ( (unsigned __int16)(v8 - 317) > 6u )
            goto LABEL_14;
LABEL_11:
          v6 = *(_QWORD *)(v4 + 32);
LABEL_12:
          v7 = *(_DWORD *)(v6 + 8);
LABEL_13:
          v60 = v7;
          sub_2205FC0((__int64)v61, v3, &v60);
          goto LABEL_14;
        }
        v10 = *(_QWORD *)(v4 + 32);
        v11 = *(_QWORD *)(v10 + 104);
        if ( v11 != 255 )
        {
          if ( v11 > 0xFE )
            goto LABEL_14;
          v7 = *(_DWORD *)(v10 + 8);
          goto LABEL_13;
        }
        v60 = *(_DWORD *)(v10 + 8);
        sub_2205FC0((__int64)v61, v3, &v60);
        v22 = *(_DWORD *)(a1 + 304);
        v23 = a1 + 280;
        if ( !v22 )
        {
          ++*(_QWORD *)(a1 + 280);
          goto LABEL_93;
        }
        v24 = *(_QWORD *)(a1 + 288);
        v25 = ((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4);
        v26 = (v22 - 1) & v25;
        v20 = (_QWORD *)(v24 + 8 * v26);
        v27 = *v20;
        if ( v4 != *v20 )
        {
          v28 = 1;
          v29 = 0;
          while ( v27 != -8 )
          {
            if ( !v29 && v27 == -16 )
              v29 = v20;
            v26 = (v22 - 1) & ((_DWORD)v26 + v28);
            v20 = (_QWORD *)(v24 + 8 * v26);
            v27 = *v20;
            if ( v4 == *v20 )
              goto LABEL_14;
            ++v28;
          }
          if ( v29 )
            v20 = v29;
          ++*(_QWORD *)(a1 + 280);
          v21 = *(_DWORD *)(a1 + 296) + 1;
          if ( 4 * v21 < 3 * v22 )
          {
            if ( v22 - *(_DWORD *)(a1 + 300) - v21 <= v22 >> 3 )
            {
              sub_1E22DE0(v23, v22);
              v30 = *(_DWORD *)(a1 + 304);
              if ( !v30 )
                goto LABEL_125;
              v31 = v30 - 1;
              v32 = *(_QWORD *)(a1 + 288);
              v33 = 0;
              v34 = v31 & v25;
              v20 = (_QWORD *)(v32 + 8LL * (v31 & v25));
              v21 = *(_DWORD *)(a1 + 296) + 1;
              v35 = 1;
              v36 = *v20;
              if ( v4 != *v20 )
              {
                while ( v36 != -8 )
                {
                  if ( !v33 && v36 == -16 )
                    v33 = v20;
                  v34 = v31 & (v35 + v34);
                  v20 = (_QWORD *)(v32 + 8LL * v34);
                  v36 = *v20;
                  if ( v4 == *v20 )
                    goto LABEL_59;
                  ++v35;
                }
                goto LABEL_73;
              }
            }
            goto LABEL_59;
          }
LABEL_93:
          sub_1E22DE0(v23, 2 * v22);
          v50 = *(_DWORD *)(a1 + 304);
          if ( !v50 )
          {
LABEL_125:
            ++*(_DWORD *)(a1 + 296);
            BUG();
          }
          v51 = v50 - 1;
          v52 = *(_QWORD *)(a1 + 288);
          v53 = v51 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v21 = *(_DWORD *)(a1 + 296) + 1;
          v20 = (_QWORD *)(v52 + 8LL * v53);
          v54 = *v20;
          if ( v4 != *v20 )
          {
            v55 = 1;
            v56 = 0;
            while ( v54 != -8 )
            {
              if ( v54 == -16 && !v56 )
                v56 = v20;
              v53 = v51 & (v55 + v53);
              v20 = (_QWORD *)(v52 + 8LL * v53);
              v54 = *v20;
              if ( v4 == *v20 )
                goto LABEL_59;
              ++v55;
            }
            if ( v56 )
              v20 = v56;
          }
          goto LABEL_59;
        }
      }
LABEL_14:
      if ( (*(_BYTE *)v4 & 4) != 0 )
      {
        v4 = *(_QWORD *)(v4 + 8);
        if ( v4 == v5 )
          goto LABEL_26;
      }
      else
      {
        while ( (*(_BYTE *)(v4 + 46) & 8) != 0 )
          v4 = *(_QWORD *)(v4 + 8);
        v4 = *(_QWORD *)(v4 + 8);
        if ( v4 == v5 )
          goto LABEL_26;
      }
    }
    if ( v8 > 0xBA6u )
    {
      if ( v8 > 0xBACu )
      {
        if ( (unsigned __int16)(v8 - 3049) > 5u )
          goto LABEL_14;
        v6 = *(_QWORD *)(v4 + 32);
        if ( *(_QWORD *)(v6 + 184) || *(_QWORD *)(v6 + 224) != 8 )
          goto LABEL_14;
        goto LABEL_12;
      }
      v12 = *(_QWORD *)(v4 + 32);
      if ( *(_QWORD *)(v12 + 304) || *(_QWORD *)(v12 + 344) != 8 )
        goto LABEL_14;
    }
    else
    {
      if ( v8 > 0xBA0u )
      {
        v9 = *(_QWORD *)(v4 + 32);
        if ( *(_QWORD *)(v9 + 224) || *(_QWORD *)(v9 + 264) != 8 )
          goto LABEL_14;
        goto LABEL_30;
      }
      if ( v8 <= 0x619u )
      {
        if ( v8 <= 0x614u )
        {
          if ( (unsigned __int16)(v8 - 1507) > 4u )
            goto LABEL_14;
          goto LABEL_11;
        }
LABEL_29:
        v9 = *(_QWORD *)(v4 + 32);
LABEL_30:
        v60 = *(_DWORD *)(v9 + 8);
        sub_2205FC0((__int64)v61, v3, &v60);
        v7 = *(_DWORD *)(*(_QWORD *)(v4 + 32) + 48LL);
        goto LABEL_13;
      }
      if ( (unsigned __int16)(v8 - 1587) > 4u )
        goto LABEL_14;
LABEL_38:
      v12 = *(_QWORD *)(v4 + 32);
    }
    v60 = *(_DWORD *)(v12 + 8);
    sub_2205FC0((__int64)v61, v3, &v60);
    v60 = *(_DWORD *)(*(_QWORD *)(v4 + 32) + 48LL);
    sub_2205FC0((__int64)v61, v3, &v60);
    v60 = *(_DWORD *)(*(_QWORD *)(v4 + 32) + 88LL);
    sub_2205FC0((__int64)v61, v3, &v60);
    v7 = *(_DWORD *)(*(_QWORD *)(v4 + 32) + 128LL);
    goto LABEL_13;
  }
  return result;
}
