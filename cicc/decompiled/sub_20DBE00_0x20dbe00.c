// Function: sub_20DBE00
// Address: 0x20dbe00
//
__int64 __fastcall sub_20DBE00(__int64 a1, int a2, unsigned int a3, __int64 a4, __int64 *a5)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 *v7; // r12
  __int64 v8; // rbx
  unsigned int v9; // edi
  __int64 *v10; // rsi
  unsigned int v11; // r8d
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // rcx
  unsigned int v15; // ecx
  __int64 *v16; // rdx
  __int64 *v17; // r10
  __int64 v18; // rsi
  unsigned __int64 v19; // r13
  _QWORD *v20; // rcx
  __int64 v21; // r14
  __int64 v22; // r13
  unsigned __int64 v23; // rbx
  unsigned __int64 v24; // r15
  __int64 v25; // rdx
  _QWORD *v26; // rax
  _QWORD *v27; // rdi
  __int64 v28; // rax
  __int64 i; // rbx
  _QWORD *v30; // rax
  _QWORD *v31; // rdi
  __int64 v32; // rax
  __int64 j; // r15
  __int16 v34; // ax
  __int16 v35; // ax
  __int64 (*v36)(); // rax
  unsigned __int64 v37; // r14
  unsigned __int64 v38; // r15
  __int64 v39; // rdx
  unsigned __int64 v40; // rsi
  unsigned __int64 v41; // rsi
  __int64 mm; // r14
  __int16 v43; // ax
  _QWORD *v44; // rax
  _QWORD *v45; // rcx
  __int64 v46; // rax
  __int64 k; // rbx
  _QWORD *v48; // rax
  _QWORD *v49; // rcx
  __int64 v50; // rax
  __int64 n; // r15
  unsigned int v53; // eax
  __int64 v54; // r13
  __m128i *v55; // rsi
  const __m128i *v56; // r9
  __m128i *v57; // rsi
  unsigned __int64 v58; // r14
  unsigned __int64 v59; // r15
  __int64 *v60; // r11
  __int64 *v61; // rbx
  unsigned int v62; // r12d
  __int16 v63; // ax
  __int64 nn; // r13
  __int64 v65; // rax
  unsigned int v66; // r10d
  __int16 v67; // ax
  _QWORD *v68; // rax
  _QWORD *v69; // rsi
  __int64 v70; // rax
  __int64 m; // r14
  unsigned __int64 v72; // rdi
  __int16 v73; // ax
  __int64 i1; // rdi
  __int64 v75; // rdx
  unsigned __int64 v76; // rdi
  __int16 v77; // ax
  __int64 v78; // rdx
  __int64 v79; // rax
  unsigned __int64 v80; // r15
  unsigned __int64 v81; // rdx
  unsigned __int64 v82; // rdx
  __int64 jj; // r15
  int v84; // eax
  int v85; // edx
  __int64 ii; // r15
  __int64 kk; // r14
  int v88; // r9d
  int v89; // r9d
  const __m128i **v90; // [rsp+10h] [rbp-B0h]
  _DWORD *v92; // [rsp+20h] [rbp-A0h]
  __int64 v93; // [rsp+28h] [rbp-98h]
  unsigned int v95; // [rsp+34h] [rbp-8Ch]
  _DWORD *v97; // [rsp+40h] [rbp-80h]
  unsigned __int64 v98; // [rsp+48h] [rbp-78h]
  __int64 *v99; // [rsp+48h] [rbp-78h]
  __int64 v100; // [rsp+50h] [rbp-70h]
  __int64 v101; // [rsp+58h] [rbp-68h]
  int v102; // [rsp+60h] [rbp-60h]
  char v103; // [rsp+67h] [rbp-59h]
  __int64 v105; // [rsp+70h] [rbp-50h]
  unsigned int v106; // [rsp+78h] [rbp-48h]
  __m128i v107; // [rsp+80h] [rbp-40h] BYREF

  v90 = (const __m128i **)(a1 + 112);
  v5 = *(_QWORD *)(a1 + 112);
  v102 = a2;
  if ( *(_QWORD *)(a1 + 120) != v5 )
    *(_QWORD *)(a1 + 120) = v5;
  v95 = 0;
  v92 = (_DWORD *)(*(_QWORD *)(a1 + 8) - 16LL);
  v97 = *(_DWORD **)a1;
  v93 = (__int64)v92;
  if ( *(_DWORD **)a1 != v92 )
  {
    v6 = *(_QWORD *)(a1 + 8) - 16LL;
    if ( *v92 == a2 )
    {
      while ( 1 )
      {
        v101 = v6 - 16;
        v105 = v6 - 16;
        if ( a2 == *(_DWORD *)(v6 - 16) )
          break;
LABEL_87:
        v6 = v101;
        if ( v97 != (_DWORD *)v101 )
        {
          v93 = v101;
          a2 = v102;
          if ( *(_DWORD *)v101 == v102 )
            continue;
        }
        return v95;
      }
      while ( 1 )
      {
        v103 = *(_BYTE *)(a1 + 136);
        v7 = *(__int64 **)(v105 + 8);
        v8 = *(_QWORD *)(v101 + 24);
        if ( *(_DWORD *)(a1 + 96) )
          break;
LABEL_11:
        v18 = *(_QWORD *)(v8 + 32);
        v19 = v8 + 24;
        v20 = v7 + 3;
        if ( v8 + 24 == v18 )
        {
          v106 = 0;
          v39 = v7[4];
          v38 = v8 + 24;
          v37 = (unsigned __int64)(v7 + 3);
          goto LABEL_35;
        }
        v106 = 0;
        v21 = *(_QWORD *)(a1 + 144);
        v98 = v8 + 24;
        v22 = *(_QWORD *)(v101 + 24);
        v23 = v8 + 24;
        v24 = (unsigned __int64)(v7 + 3);
        while ( 1 )
        {
          v25 = v7[4];
          if ( v25 == v24 )
            break;
          v26 = (_QWORD *)(*(_QWORD *)v23 & 0xFFFFFFFFFFFFFFF8LL);
          v27 = v26;
          if ( !v26 )
            BUG();
          v23 = *(_QWORD *)v23 & 0xFFFFFFFFFFFFFFF8LL;
          v28 = *v26;
          if ( (v28 & 4) == 0 && (*((_BYTE *)v27 + 46) & 4) != 0 )
          {
            for ( i = v28; ; i = *(_QWORD *)v23 )
            {
              v23 = i & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_BYTE *)(v23 + 46) & 4) == 0 )
                break;
            }
          }
          v30 = (_QWORD *)(*(_QWORD *)v24 & 0xFFFFFFFFFFFFFFF8LL);
          v31 = v30;
          if ( !v30 )
            BUG();
          v24 = *(_QWORD *)v24 & 0xFFFFFFFFFFFFFFF8LL;
          v32 = *v30;
          if ( (v32 & 4) == 0 && (*((_BYTE *)v31 + 46) & 4) != 0 )
          {
            for ( j = v32; ; j = *(_QWORD *)v24 )
            {
              v24 = j & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_BYTE *)(v24 + 46) & 4) == 0 )
                break;
            }
          }
          v34 = **(_WORD **)(v23 + 16);
          if ( v34 == 12 || v34 == 2 )
          {
            while ( v23 != v18 )
            {
              v44 = (_QWORD *)(*(_QWORD *)v23 & 0xFFFFFFFFFFFFFFF8LL);
              v45 = v44;
              if ( !v44 )
                BUG();
              v23 = *(_QWORD *)v23 & 0xFFFFFFFFFFFFFFF8LL;
              v46 = *v44;
              if ( (v46 & 4) == 0 && (*((_BYTE *)v45 + 46) & 4) != 0 )
              {
                for ( k = v46; ; k = *(_QWORD *)v23 )
                {
                  v23 = k & 0xFFFFFFFFFFFFFFF8LL;
                  if ( (*(_BYTE *)(v23 + 46) & 4) == 0 )
                    break;
                }
              }
              v43 = **(_WORD **)(v23 + 16);
              if ( v43 != 2 && v43 != 12 )
                goto LABEL_28;
            }
            v37 = v24;
            v20 = v7 + 3;
            v38 = v23;
            v8 = v22;
            v19 = v98;
            while ( 1 )
            {
              v67 = **(_WORD **)(v37 + 16);
              if ( v67 != 2 && v67 != 12 )
                break;
              if ( v25 == v37 )
                goto LABEL_51;
              v68 = (_QWORD *)(*(_QWORD *)v37 & 0xFFFFFFFFFFFFFFF8LL);
              v69 = v68;
              if ( !v68 )
                BUG();
              v37 = *(_QWORD *)v37 & 0xFFFFFFFFFFFFFFF8LL;
              v70 = *v68;
              if ( (v70 & 4) == 0 && (*((_BYTE *)v69 + 46) & 4) != 0 )
              {
                for ( m = v70; ; m = *(_QWORD *)v37 )
                {
                  v37 = m & 0xFFFFFFFFFFFFFFF8LL;
                  if ( (*(_BYTE *)(v37 + 46) & 4) == 0 )
                    break;
                }
              }
            }
            if ( (*(_BYTE *)v37 & 4) == 0 )
            {
              while ( (*(_BYTE *)(v37 + 46) & 8) != 0 )
                v37 = *(_QWORD *)(v37 + 8);
            }
            goto LABEL_50;
          }
LABEL_28:
          while ( 1 )
          {
            v35 = **(_WORD **)(v24 + 16);
            if ( v35 != 12 && v35 != 2 )
              break;
            if ( v25 == v24 )
            {
              v37 = v24;
              v38 = v23;
              v20 = v7 + 3;
              v8 = v22;
              v19 = v98;
              if ( (*(_BYTE *)v38 & 4) == 0 && (*(_BYTE *)(v38 + 46) & 8) != 0 )
              {
                do
                  v38 = *(_QWORD *)(v38 + 8);
                while ( (*(_BYTE *)(v38 + 46) & 8) != 0 );
              }
              goto LABEL_177;
            }
            v48 = (_QWORD *)(*(_QWORD *)v24 & 0xFFFFFFFFFFFFFFF8LL);
            v49 = v48;
            if ( !v48 )
              BUG();
            v24 = *(_QWORD *)v24 & 0xFFFFFFFFFFFFFFF8LL;
            v50 = *v48;
            if ( (v50 & 4) == 0 && (*((_BYTE *)v49 + 46) & 4) != 0 )
            {
              for ( n = v50; ; n = *(_QWORD *)v24 )
              {
                v24 = n & 0xFFFFFFFFFFFFFFF8LL;
                if ( (*(_BYTE *)(v24 + 46) & 4) == 0 )
                  break;
              }
            }
          }
          if ( !(unsigned __int8)sub_1E15D60(v23, v24, 0) || **(_WORD **)(v23 + 16) == 1 )
          {
            v58 = v24;
            v80 = v23;
            v20 = v7 + 3;
            v8 = v22;
            v19 = v98;
            if ( (*(_BYTE *)v80 & 4) == 0 )
            {
              while ( (*(_BYTE *)(v80 + 46) & 8) != 0 )
                v80 = *(_QWORD *)(v80 + 8);
            }
            v38 = *(_QWORD *)(v80 + 8);
            if ( (*(_BYTE *)v58 & 4) == 0 )
            {
              while ( (*(_BYTE *)(v58 + 46) & 8) != 0 )
                v58 = *(_QWORD *)(v58 + 8);
            }
LABEL_126:
            v18 = *(_QWORD *)(v8 + 32);
            v37 = *(_QWORD *)(v58 + 8);
            v39 = v7[4];
            if ( v38 != v18 )
            {
              if ( v39 == v37 )
              {
LABEL_222:
                v81 = *(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL;
                if ( !v81 )
                  BUG();
                v38 = *(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL;
                if ( (*(_QWORD *)v81 & 4) != 0 )
                {
                  if ( (unsigned __int16)(**(_WORD **)(v81 + 16) - 12) > 1u )
                    goto LABEL_177;
                  goto LABEL_225;
                }
                if ( (*(_BYTE *)(v81 + 46) & 4) != 0 )
                {
                  for ( ii = *(_QWORD *)v81; ; ii = *(_QWORD *)v38 )
                  {
                    v38 = ii & 0xFFFFFFFFFFFFFFF8LL;
                    if ( (*(_BYTE *)(v38 + 46) & 4) == 0 )
                      break;
                  }
                  goto LABEL_232;
                }
                if ( (unsigned __int16)(**(_WORD **)(v81 + 16) - 12) > 1u )
                  goto LABEL_236;
                do
                {
LABEL_225:
                  if ( v38 == v18 )
                    goto LABEL_90;
                  v82 = *(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( !v82 )
                    BUG();
                  v38 = *(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( (*(_QWORD *)v82 & 4) == 0 && (*(_BYTE *)(v82 + 46) & 4) != 0 )
                  {
                    for ( jj = *(_QWORD *)v82; ; jj = *(_QWORD *)v38 )
                    {
                      v38 = jj & 0xFFFFFFFFFFFFFFF8LL;
                      if ( (*(_BYTE *)(v38 + 46) & 4) == 0 )
                        break;
                    }
                  }
LABEL_232:
                  ;
                }
                while ( (unsigned __int16)(**(_WORD **)(v38 + 16) - 12) <= 1u );
                if ( (*(_BYTE *)v38 & 4) != 0 )
                {
LABEL_177:
                  v38 = *(_QWORD *)(v38 + 8);
                }
                else
                {
LABEL_236:
                  while ( (*(_BYTE *)(v38 + 46) & 8) != 0 )
                    v38 = *(_QWORD *)(v38 + 8);
                  v38 = *(_QWORD *)(v38 + 8);
                }
              }
              goto LABEL_51;
            }
LABEL_35:
            if ( v39 == v37 )
              goto LABEL_51;
            v40 = *(_QWORD *)v37 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v40 )
              BUG();
            v37 = *(_QWORD *)v37 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_QWORD *)v40 & 4) != 0 )
            {
              if ( (unsigned __int16)(**(_WORD **)(v40 + 16) - 12) <= 1u )
                goto LABEL_39;
              goto LABEL_50;
            }
            if ( (*(_BYTE *)(v40 + 46) & 4) != 0 )
            {
              for ( kk = *(_QWORD *)v40; ; kk = *(_QWORD *)v37 )
              {
                v37 = kk & 0xFFFFFFFFFFFFFFF8LL;
                if ( (*(_BYTE *)(v37 + 46) & 4) == 0 )
                  break;
              }
              goto LABEL_46;
            }
            if ( (unsigned __int16)(**(_WORD **)(v40 + 16) - 12) <= 1u )
            {
              do
              {
LABEL_39:
                if ( v39 == v37 )
                  goto LABEL_90;
                v41 = *(_QWORD *)v37 & 0xFFFFFFFFFFFFFFF8LL;
                if ( !v41 )
                  BUG();
                v37 = *(_QWORD *)v37 & 0xFFFFFFFFFFFFFFF8LL;
                if ( (*(_QWORD *)v41 & 4) == 0 && (*(_BYTE *)(v41 + 46) & 4) != 0 )
                {
                  for ( mm = *(_QWORD *)v41; ; mm = *(_QWORD *)v37 )
                  {
                    v37 = mm & 0xFFFFFFFFFFFFFFF8LL;
                    if ( (*(_BYTE *)(v37 + 46) & 4) == 0 )
                      break;
                  }
                }
LABEL_46:
                ;
              }
              while ( (unsigned __int16)(**(_WORD **)(v37 + 16) - 12) <= 1u );
              if ( (*(_BYTE *)v37 & 4) != 0 )
              {
LABEL_50:
                v37 = *(_QWORD *)(v37 + 8);
                goto LABEL_51;
              }
            }
            for ( ; (*(_BYTE *)(v37 + 46) & 8) != 0; v37 = *(_QWORD *)(v37 + 8) )
              ;
            goto LABEL_50;
          }
          v36 = *(__int64 (**)())(*(_QWORD *)v21 + 192LL);
          if ( v36 != sub_1F39430 && !((unsigned __int8 (__fastcall *)(__int64, unsigned __int64))v36)(v21, v23) )
          {
            v58 = v24;
            v59 = v23;
            v20 = v7 + 3;
            v8 = v22;
            v19 = v98;
            if ( (*(_BYTE *)v59 & 4) == 0 && (*(_BYTE *)(v59 + 46) & 8) != 0 )
            {
              do
                v59 = *(_QWORD *)(v59 + 8);
              while ( (*(_BYTE *)(v59 + 46) & 8) != 0 );
            }
            v38 = *(_QWORD *)(v59 + 8);
            if ( (*(_BYTE *)v58 & 4) == 0 )
            {
              while ( (*(_BYTE *)(v58 + 46) & 8) != 0 )
                v58 = *(_QWORD *)(v58 + 8);
            }
            goto LABEL_126;
          }
          v18 = *(_QWORD *)(v22 + 32);
          ++v106;
          if ( v18 == v23 )
          {
            v37 = v24;
            v20 = v7 + 3;
            v38 = v23;
            v39 = v7[4];
            v8 = v22;
            v19 = v98;
            goto LABEL_35;
          }
        }
        v37 = v24;
        v38 = v23;
        v20 = v7 + 3;
        v8 = v22;
        v19 = v98;
        if ( v18 != v38 )
          goto LABEL_222;
LABEL_51:
        while ( v19 != v38 )
        {
          if ( **(_WORD **)(v38 + 16) != 2 )
            break;
          if ( (*(_BYTE *)v38 & 4) == 0 && (*(_BYTE *)(v38 + 46) & 8) != 0 )
          {
            do
              v38 = *(_QWORD *)(v38 + 8);
            while ( (*(_BYTE *)(v38 + 46) & 8) != 0 );
          }
          v38 = *(_QWORD *)(v38 + 8);
        }
        while ( v20 != (_QWORD *)v37 && **(_WORD **)(v37 + 16) == 2 )
        {
          if ( (*(_BYTE *)v37 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v37 + 46) & 8) != 0 )
              v37 = *(_QWORD *)(v37 + 8);
          }
          v37 = *(_QWORD *)(v37 + 8);
        }
LABEL_90:
        if ( !v106 )
          goto LABEL_85;
        if ( (a5 == (__int64 *)v8 || a5 == v7)
          && (!v103 || (unsigned int)((__int64)(*(_QWORD *)(v8 + 96) - *(_QWORD *)(v8 + 88)) >> 3) == 1) )
        {
          v60 = v7;
          if ( a5 == (__int64 *)v8 )
            v19 = (unsigned __int64)v20;
          else
            v60 = (__int64 *)v8;
          if ( v19 == v60[4] )
            goto LABEL_108;
          v100 = v8;
          v61 = v60;
          v99 = v7;
          v62 = 0;
          while ( 1 )
          {
            v19 = *(_QWORD *)v19 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v19 )
              BUG();
            v63 = *(_WORD *)(v19 + 46);
            if ( (*(_QWORD *)v19 & 4) != 0 )
            {
              if ( (v63 & 4) != 0 )
                goto LABEL_156;
            }
            else if ( (v63 & 4) != 0 )
            {
              for ( nn = *(_QWORD *)v19; ; nn = *(_QWORD *)v19 )
              {
                v19 = nn & 0xFFFFFFFFFFFFFFF8LL;
                v63 = *(_WORD *)(v19 + 46);
                if ( (v63 & 4) == 0 )
                  break;
              }
            }
            if ( (v63 & 8) != 0 )
            {
              LOBYTE(v65) = sub_1E15D00(v19, 0x40u, 1);
              goto LABEL_147;
            }
LABEL_156:
            v65 = (*(_QWORD *)(*(_QWORD *)(v19 + 16) + 8LL) >> 6) & 1LL;
LABEL_147:
            if ( (_BYTE)v65 )
            {
              ++v62;
              if ( v19 != v61[4] )
                continue;
            }
            v66 = v62;
            v8 = v100;
            v7 = v99;
            if ( v106 <= v66 )
            {
              if ( *(_QWORD *)(v100 + 32) == v38 )
                goto LABEL_151;
              goto LABEL_94;
            }
            goto LABEL_108;
          }
        }
        if ( *(_QWORD *)(v8 + 32) != v38 )
          goto LABEL_94;
LABEL_151:
        if ( v7[4] == v37 && (unsigned __int8)sub_20D6880((_QWORD *)v8) && (unsigned __int8)sub_20D6880(v7) )
          goto LABEL_108;
LABEL_94:
        if ( sub_1DD69A0(v8, (__int64)v7) && v7[4] == v37 )
          goto LABEL_108;
        if ( sub_1DD69A0((__int64)v7, v8) )
        {
          if ( *(_QWORD *)(v8 + 32) == v38 )
            goto LABEL_108;
          if ( v103 )
          {
LABEL_99:
            if ( a5 != (__int64 *)v8 && a4 != 0 && a5 != v7 )
            {
              if ( (unsigned int)((__int64)(*(_QWORD *)(v8 + 96) - *(_QWORD *)(v8 + 88)) >> 3) != 1 )
              {
                v53 = v106;
LABEL_103:
                if ( a3 <= v53 )
                  goto LABEL_108;
                if ( v53 > 1 )
                {
                  v54 = **(_QWORD **)(v8 + 56) + 112LL;
                  if ( ((unsigned __int8)sub_1560180(v54, 34) || (unsigned __int8)sub_1560180(v54, 17))
                    && (*(_QWORD *)(v8 + 32) == v38 || v7[4] == v37) )
                  {
                    goto LABEL_108;
                  }
                }
                goto LABEL_85;
              }
LABEL_190:
              v72 = *(_QWORD *)(v8 + 24) & 0xFFFFFFFFFFFFFFF8LL;
              if ( !v72 )
                BUG();
              v73 = *(_WORD *)(v72 + 46);
              if ( (*(_QWORD *)v72 & 4) != 0 )
              {
                if ( (v73 & 4) == 0 )
                  goto LABEL_196;
              }
              else
              {
                if ( (v73 & 4) != 0 )
                {
                  for ( i1 = *(_QWORD *)v72; ; i1 = *(_QWORD *)v72 )
                  {
                    v72 = i1 & 0xFFFFFFFFFFFFFFF8LL;
                    v73 = *(_WORD *)(v72 + 46);
                    if ( (v73 & 4) == 0 )
                      break;
                  }
                }
LABEL_196:
                if ( (v73 & 8) != 0 )
                {
                  LOBYTE(v75) = sub_1E15D00(v72, 0x20u, 1);
                  goto LABEL_198;
                }
              }
              v75 = (*(_QWORD *)(*(_QWORD *)(v72 + 16) + 8LL) >> 5) & 1LL;
LABEL_198:
              v53 = v106;
              if ( (_BYTE)v75 )
                goto LABEL_103;
              v76 = v7[3] & 0xFFFFFFFFFFFFFFF8LL;
              if ( !v76 )
                BUG();
              v77 = *(_WORD *)(v76 + 46);
              v78 = *(_QWORD *)v76;
              if ( (*(_QWORD *)v76 & 4) == 0 )
              {
                if ( (v77 & 4) != 0 )
                {
                  while ( 1 )
                  {
                    v76 = v78 & 0xFFFFFFFFFFFFFFF8LL;
                    v77 = *(_WORD *)((v78 & 0xFFFFFFFFFFFFFFF8LL) + 46);
                    if ( (v77 & 4) == 0 )
                      break;
                    v78 = *(_QWORD *)v76;
                  }
                }
LABEL_205:
                if ( (v77 & 8) != 0 )
                  LOBYTE(v79) = sub_1E15D00(v76, 0x20u, 1);
                else
LABEL_206:
                  v79 = (*(_QWORD *)(*(_QWORD *)(v76 + 16) + 8LL) >> 5) & 1LL;
                v53 = ((_BYTE)v79 == 0) + v106;
                goto LABEL_103;
              }
              if ( (v77 & 4) == 0 )
                goto LABEL_205;
              goto LABEL_206;
            }
          }
          else
          {
LABEL_188:
            if ( a4 != 0 && a5 != (__int64 *)v8 && a5 != v7 )
              goto LABEL_190;
          }
          v53 = v106;
          goto LABEL_103;
        }
        if ( !v103 )
          goto LABEL_188;
        if ( *(_QWORD *)(v8 + 32) != v38 || v7[4] != v37 )
          goto LABEL_99;
        if ( (unsigned int)((__int64)(*(_QWORD *)(v8 + 96) - *(_QWORD *)(v8 + 88)) >> 3) )
        {
          if ( !sub_1DD6C00((__int64 *)v8) || v8 == *(_QWORD *)(*(_QWORD *)(v8 + 56) + 328LL) )
            goto LABEL_108;
        }
        else if ( v8 == *(_QWORD *)(*(_QWORD *)(v8 + 56) + 328LL) )
        {
          goto LABEL_108;
        }
        if ( sub_1DD6C00((__int64 *)(*(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL))
          && (!(unsigned int)((v7[12] - v7[11]) >> 3) || sub_1DD6C00(v7))
          && v7 != *(__int64 **)(v7[7] + 328)
          && sub_1DD6C00((__int64 *)(*v7 & 0xFFFFFFFFFFFFFFF8LL)) )
        {
          goto LABEL_99;
        }
LABEL_108:
        if ( v95 < v106 )
        {
          v55 = *(__m128i **)(a1 + 112);
          if ( v55 != *(__m128i **)(a1 + 120) )
            *(_QWORD *)(a1 + 120) = v55;
          v107.m128i_i64[1] = v38;
          v107.m128i_i64[0] = v93;
          v56 = *(const __m128i **)(a1 + 128);
          if ( v56 == v55 )
          {
            sub_20DBC80(v90, v55, &v107);
            v57 = *(__m128i **)(a1 + 120);
            v56 = *(const __m128i **)(a1 + 128);
          }
          else
          {
            if ( v55 )
            {
              *v55 = _mm_loadu_si128(&v107);
              v55 = *(__m128i **)(a1 + 120);
              v56 = *(const __m128i **)(a1 + 128);
            }
            v57 = v55 + 1;
            *(_QWORD *)(a1 + 120) = v57;
          }
          v92 = (_DWORD *)v93;
          goto LABEL_116;
        }
        if ( v92 == (_DWORD *)(v101 + 16) && v95 == v106 )
        {
          v57 = *(__m128i **)(a1 + 120);
          v56 = *(const __m128i **)(a1 + 128);
LABEL_116:
          v107.m128i_i64[1] = v37;
          v107.m128i_i64[0] = v105;
          if ( v56 == v57 )
          {
            sub_20DBC80(v90, v56, &v107);
          }
          else
          {
            if ( v57 )
            {
              *v57 = _mm_loadu_si128(&v107);
              v57 = *(__m128i **)(a1 + 120);
            }
            *(_QWORD *)(a1 + 120) = v57 + 1;
          }
          v95 = v106;
        }
LABEL_85:
        if ( v97 != (_DWORD *)v105 )
        {
          v105 -= 16;
          if ( v102 == *(_DWORD *)v105 )
            continue;
        }
        goto LABEL_87;
      }
      v9 = *(_DWORD *)(a1 + 104);
      v10 = *(__int64 **)(a1 + 88);
      if ( v9 )
      {
        v11 = v9 - 1;
        v12 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v13 = &v10[2 * v12];
        v14 = *v13;
        if ( v8 != *v13 )
        {
          v84 = 1;
          while ( v14 != -8 )
          {
            v89 = v84 + 1;
            v12 = v11 & (v84 + v12);
            v13 = &v10[2 * v12];
            v14 = *v13;
            if ( v8 == *v13 )
              goto LABEL_9;
            v84 = v89;
          }
          v13 = &v10[2 * v9];
        }
LABEL_9:
        v15 = v11 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v16 = &v10[2 * v15];
        v17 = (__int64 *)*v16;
        if ( v7 == (__int64 *)*v16 )
        {
LABEL_10:
          if ( *((_DWORD *)v13 + 2) != *((_DWORD *)v16 + 2) )
            goto LABEL_85;
          goto LABEL_11;
        }
        v85 = 1;
        while ( v17 != (__int64 *)-8LL )
        {
          v88 = v85 + 1;
          v15 = v11 & (v85 + v15);
          v16 = &v10[2 * v15];
          v17 = (__int64 *)*v16;
          if ( v7 == (__int64 *)*v16 )
            goto LABEL_10;
          v85 = v88;
        }
        v16 = &v10[2 * v9];
        v10 = v13;
      }
      else
      {
        v16 = *(__int64 **)(a1 + 88);
      }
      v13 = v10;
      goto LABEL_10;
    }
  }
  return v95;
}
