// Function: sub_34C2FF0
// Address: 0x34c2ff0
//
__int64 __fastcall sub_34C2FF0(_QWORD *a1, int a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  _QWORD *v6; // r14
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // r15
  unsigned int v10; // edi
  __int64 *v11; // rsi
  unsigned int v12; // r11d
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // rcx
  unsigned int v16; // ecx
  __int64 *v17; // rdx
  __int64 v18; // rbx
  unsigned __int64 v19; // r11
  unsigned __int64 v20; // rbx
  __int64 v21; // r13
  unsigned __int64 v22; // r15
  __int64 v23; // r14
  unsigned __int64 v24; // r12
  _QWORD *v25; // rax
  _QWORD *v26; // rdx
  __int64 v27; // rax
  __int64 i; // rbx
  __int16 v29; // ax
  _QWORD *v30; // rax
  _QWORD *v31; // rdx
  __int64 v32; // rax
  __int64 j; // r12
  __int16 v34; // ax
  bool v35; // al
  __int64 v36; // r12
  unsigned __int64 v37; // rcx
  __int64 v38; // r15
  __int64 v39; // rdx
  unsigned __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rsi
  char v44; // al
  unsigned int v45; // ebx
  __int64 v46; // r8
  unsigned __int64 v47; // rdi
  unsigned int v48; // r12d
  __int64 v49; // rbx
  int v50; // esi
  int v51; // edx
  __int64 k; // rdi
  __int64 v53; // rax
  unsigned int v54; // ebx
  __m128i *v55; // rax
  const __m128i *v56; // rsi
  __m128i *v57; // rax
  unsigned __int64 v58; // rdi
  __int64 v59; // rdx
  int v60; // ecx
  __int64 v61; // rax
  unsigned __int64 v62; // rdi
  __int64 v63; // rdx
  int v64; // ecx
  __int64 v65; // rax
  int v66; // eax
  int v67; // edx
  int v69; // r8d
  int v70; // r8d
  unsigned __int64 v71; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v72; // [rsp+10h] [rbp-E0h]
  __int64 v73; // [rsp+18h] [rbp-D8h]
  unsigned __int64 *v76; // [rsp+38h] [rbp-B8h]
  _DWORD *v78; // [rsp+48h] [rbp-A8h]
  __int64 v79; // [rsp+50h] [rbp-A0h]
  __int64 v80; // [rsp+58h] [rbp-98h]
  unsigned int v81; // [rsp+60h] [rbp-90h]
  __int64 v83; // [rsp+68h] [rbp-88h]
  _DWORD *v84; // [rsp+70h] [rbp-80h]
  __int64 **v85; // [rsp+78h] [rbp-78h]
  __int64 v86; // [rsp+80h] [rbp-70h]
  _DWORD *v87; // [rsp+88h] [rbp-68h]
  unsigned __int64 v88; // [rsp+90h] [rbp-60h]
  unsigned __int64 v89; // [rsp+90h] [rbp-60h]
  _QWORD *v90; // [rsp+98h] [rbp-58h]
  bool v91; // [rsp+98h] [rbp-58h]
  __int64 v92; // [rsp+A0h] [rbp-50h]
  bool v93; // [rsp+A0h] [rbp-50h]
  unsigned __int8 v94; // [rsp+ABh] [rbp-45h]
  unsigned int v95; // [rsp+ACh] [rbp-44h]
  __m128i v96; // [rsp+B0h] [rbp-40h] BYREF

  v76 = a1 + 13;
  v5 = a1[13];
  if ( a1[14] != v5 )
    a1[14] = v5;
  v84 = (_DWORD *)*a1;
  v79 = a1[1] - 24LL;
  if ( *a1 != v79 )
  {
    v6 = a1;
    v80 = 0;
    v83 = 0;
    v78 = (_DWORD *)(a1[1] - 24LL);
    v81 = 0;
    while ( 1 )
    {
      if ( *v78 != a2 )
        return v81;
      v87 = v78 - 6;
      v7 = (__int64)(v78 - 6);
      if ( a2 != *(v78 - 6) )
        goto LABEL_69;
      do
      {
        v8 = *(_QWORD *)(v7 + 8);
        v86 = v6[31];
        v85 = (__int64 **)v6[29];
        v94 = *((_BYTE *)v6 + 128);
        v9 = *((_QWORD *)v87 + 4);
        if ( !*((_DWORD *)v6 + 22) )
          goto LABEL_12;
        v10 = *((_DWORD *)v6 + 24);
        v11 = (__int64 *)v6[10];
        if ( v10 )
        {
          v12 = v10 - 1;
          v13 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v14 = &v11[2 * v13];
          v15 = *v14;
          if ( v9 != *v14 )
          {
            v66 = 1;
            while ( v15 != -4096 )
            {
              v70 = v66 + 1;
              v13 = v12 & (v66 + v13);
              v14 = &v11[2 * v13];
              v15 = *v14;
              if ( v9 == *v14 )
                goto LABEL_10;
              v66 = v70;
            }
            v14 = &v11[2 * v10];
          }
LABEL_10:
          v16 = v12 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v17 = &v11[2 * v16];
          v18 = *v17;
          if ( v8 == *v17 )
            goto LABEL_11;
          v67 = 1;
          while ( v18 != -4096 )
          {
            v69 = v67 + 1;
            v16 = v12 & (v16 + v67);
            v17 = &v11[2 * v16];
            v18 = *v17;
            if ( v8 == *v17 )
              goto LABEL_11;
            v67 = v69;
          }
          v17 = &v11[2 * v10];
          v11 = v14;
        }
        else
        {
          v17 = (__int64 *)v6[10];
        }
        v14 = v11;
LABEL_11:
        if ( *((_DWORD *)v14 + 2) != *((_DWORD *)v17 + 2) )
          goto LABEL_67;
LABEL_12:
        v95 = 0;
        v19 = v8 + 48;
        v92 = v7;
        v20 = v9 + 48;
        v21 = *((_QWORD *)v87 + 4);
        v22 = v9 + 48;
        v90 = v6;
        v23 = v8;
        v24 = v8 + 48;
        do
        {
LABEL_13:
          if ( *(_QWORD *)(v21 + 56) == v20 )
          {
            v20 = v22;
            break;
          }
          v25 = (_QWORD *)(*(_QWORD *)v20 & 0xFFFFFFFFFFFFFFF8LL);
          v26 = v25;
          if ( !v25 )
            BUG();
          v20 = *(_QWORD *)v20 & 0xFFFFFFFFFFFFFFF8LL;
          v27 = *v25;
          if ( (v27 & 4) == 0 && (*((_BYTE *)v26 + 44) & 4) != 0 )
          {
            for ( i = v27; ; i = *(_QWORD *)v20 )
            {
              v20 = i & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_BYTE *)(v20 + 44) & 4) == 0 )
                break;
            }
          }
          v29 = *(_WORD *)(v20 + 68);
        }
        while ( (unsigned __int16)(v29 - 14) <= 4u || v29 == 3 );
        while ( v24 != *(_QWORD *)(v23 + 56) )
        {
          v30 = (_QWORD *)(*(_QWORD *)v24 & 0xFFFFFFFFFFFFFFF8LL);
          v31 = v30;
          if ( !v30 )
            BUG();
          v24 = *(_QWORD *)v24 & 0xFFFFFFFFFFFFFFF8LL;
          v32 = *v30;
          if ( (v32 & 4) == 0 && (*((_BYTE *)v31 + 44) & 4) != 0 )
          {
            for ( j = v32; ; j = *(_QWORD *)v24 )
            {
              v24 = j & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_BYTE *)(v24 + 44) & 4) == 0 )
                break;
            }
          }
          v34 = *(_WORD *)(v24 + 68);
          if ( (unsigned __int16)(v34 - 14) > 4u && v34 != 3 )
          {
            if ( v19 != v24 && v22 != v20 )
            {
              v88 = v19;
              v35 = sub_2E88AF0(v20, v24, 0);
              v19 = v88;
              if ( v35
                && (unsigned int)*(unsigned __int16 *)(v20 + 68) - 1 > 1
                && *(char *)(v20 + 45) >= 0
                && *(char *)(v24 + 45) >= 0 )
              {
                ++v95;
                v80 = v24;
                v83 = v20;
                goto LABEL_13;
              }
            }
            break;
          }
        }
        v36 = v23;
        v37 = v22;
        v6 = v90;
        v38 = v21;
        v7 = v92;
        if ( !v95 )
          goto LABEL_67;
        v39 = *(_QWORD *)(v38 + 56);
        if ( v39 == v37 )
        {
          v40 = v37;
        }
        else
        {
          v40 = *(_QWORD *)(v38 + 56);
          do
          {
            if ( (unsigned __int16)(*(_WORD *)(v40 + 68) - 14) > 4u )
              break;
            if ( (*(_BYTE *)v40 & 4) == 0 )
            {
              while ( (*(_BYTE *)(v40 + 44) & 8) != 0 )
                v40 = *(_QWORD *)(v40 + 8);
            }
            v40 = *(_QWORD *)(v40 + 8);
          }
          while ( v37 != v40 );
        }
        if ( v83 == v40 )
        {
          v41 = *(_QWORD *)(v36 + 56);
          v83 = *(_QWORD *)(v38 + 56);
          v91 = 1;
          if ( v41 == v19 )
          {
LABEL_123:
            v43 = v80;
            if ( v80 == v19 )
              goto LABEL_124;
            goto LABEL_48;
          }
        }
        else
        {
          v41 = *(_QWORD *)(v36 + 56);
          v91 = v83 == v39;
          if ( v41 == v19 )
            goto LABEL_123;
        }
        v42 = v41;
        do
        {
          if ( (unsigned __int16)(*(_WORD *)(v42 + 68) - 14) > 4u )
            break;
          if ( (*(_BYTE *)v42 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v42 + 44) & 8) != 0 )
              v42 = *(_QWORD *)(v42 + 8);
          }
          v42 = *(_QWORD *)(v42 + 8);
        }
        while ( v19 != v42 );
        v43 = v80;
        if ( v80 == v42 )
        {
LABEL_124:
          v80 = v41;
          v93 = 1;
          goto LABEL_49;
        }
LABEL_48:
        v93 = v41 == v43;
LABEL_49:
        if ( a5 != v38 && a5 != v36 || v94 && *(_DWORD *)(v38 + 120) != 1 )
          goto LABEL_51;
        v46 = v36;
        if ( a5 == v38 )
          v39 = v41;
        v47 = v19;
        if ( a5 != v38 )
        {
          v46 = v38;
          v47 = v37;
        }
        if ( v39 == v47 )
          goto LABEL_100;
        v72 = v37;
        v71 = v19;
        v73 = v36;
        v48 = 0;
        v49 = v46;
        do
        {
          v47 = *(_QWORD *)v47 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v47 )
            BUG();
          v50 = *(_DWORD *)(v47 + 44);
          LOBYTE(v51) = v50;
          if ( (*(_QWORD *)v47 & 4) != 0 )
          {
            if ( (v50 & 4) != 0 )
            {
LABEL_120:
              v53 = (*(_QWORD *)(*(_QWORD *)(v47 + 16) + 24LL) >> 9) & 1LL;
              goto LABEL_97;
            }
          }
          else if ( (v50 & 4) != 0 )
          {
            for ( k = *(_QWORD *)v47; ; k = *(_QWORD *)v47 )
            {
              v47 = k & 0xFFFFFFFFFFFFFFF8LL;
              v51 = *(_DWORD *)(v47 + 44) & 0xFFFFFF;
              if ( (*(_DWORD *)(v47 + 44) & 4) == 0 )
                break;
            }
          }
          if ( (v51 & 8) == 0 )
            goto LABEL_120;
          LOBYTE(v53) = sub_2E88A90(v47, 512, 1);
LABEL_97:
          if ( !(_BYTE)v53 )
            break;
          ++v48;
        }
        while ( v47 != *(_QWORD *)(v49 + 56) );
        v54 = v48;
        v37 = v72;
        v19 = v71;
        v36 = v73;
        if ( v95 > v54 )
          goto LABEL_100;
LABEL_51:
        if ( v93 && v91 )
        {
          if ( *(_DWORD *)(v38 + 120)
            || v37 != (*(_QWORD *)(v38 + 48) & 0xFFFFFFFFFFFFFFF8LL)
            && (v89 = v19, v44 = sub_34BE9C0(v38), v19 = v89, !v44)
            || *(_DWORD *)(v36 + 120)
            || v19 != (*(_QWORD *)(v36 + 48) & 0xFFFFFFFFFFFFFFF8LL) && !(unsigned __int8)sub_34BE9C0(v36) )
          {
            if ( !sub_2E322F0(v38, v36) )
            {
LABEL_57:
              if ( sub_2E322F0(v36, v38) && v91
                || (v91 & v94) != 0
                && v93
                && (*(_DWORD *)(v38 + 120) && !sub_2E32580((__int64 *)v38)
                 || v38 == *(_QWORD *)(*(_QWORD *)(v38 + 32) + 328LL)
                 || !sub_2E32580((__int64 *)(*(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL))
                 || *(_DWORD *)(v36 + 120) && !sub_2E32580((__int64 *)v36)
                 || v36 == *(_QWORD *)(*(_QWORD *)(v36 + 32) + 328LL)
                 || !sub_2E32580((__int64 *)(*(_QWORD *)v36 & 0xFFFFFFFFFFFFFFF8LL))) )
              {
                goto LABEL_100;
              }
              if ( a5 == v38 || a4 == 0 || a5 == v36 )
              {
                v45 = v95;
              }
              else
              {
                if ( *(_DWORD *)(v38 + 120) != 1 )
                {
                  v45 = v95;
                  if ( v94 )
                    goto LABEL_65;
                }
                v58 = *(_QWORD *)(v38 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                if ( !v58 )
                  BUG();
                v59 = *(_QWORD *)v58;
                v60 = *(_DWORD *)(v58 + 44);
                if ( (*(_QWORD *)v58 & 4) != 0 )
                {
                  if ( (v60 & 4) == 0 )
                    goto LABEL_137;
                }
                else
                {
                  if ( (v60 & 4) != 0 )
                  {
                    while ( 1 )
                    {
                      v58 = v59 & 0xFFFFFFFFFFFFFFF8LL;
                      v60 = *(_DWORD *)((v59 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 0xFFFFFF;
                      if ( (*(_DWORD *)((v59 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
                        break;
                      v59 = *(_QWORD *)v58;
                    }
                  }
LABEL_137:
                  if ( (v60 & 8) != 0 )
                  {
                    LOBYTE(v61) = sub_2E88A90(v58, 256, 1);
                    goto LABEL_139;
                  }
                }
                v61 = (*(_QWORD *)(*(_QWORD *)(v58 + 16) + 24LL) >> 8) & 1LL;
LABEL_139:
                v45 = v95;
                if ( !(_BYTE)v61 )
                {
                  v62 = *(_QWORD *)(v36 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( !v62 )
                    BUG();
                  v63 = *(_QWORD *)v62;
                  v64 = *(_DWORD *)(v62 + 44);
                  if ( (*(_QWORD *)v62 & 4) != 0 )
                  {
                    if ( (v64 & 4) != 0 )
                      goto LABEL_143;
                  }
                  else if ( (v64 & 4) != 0 )
                  {
                    while ( 1 )
                    {
                      v62 = v63 & 0xFFFFFFFFFFFFFFF8LL;
                      v64 = *(_DWORD *)((v63 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 0xFFFFFF;
                      if ( (*(_DWORD *)((v63 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
                        break;
                      v63 = *(_QWORD *)v62;
                    }
                  }
                  if ( (v64 & 8) != 0 )
                    LOBYTE(v65) = sub_2E88A90(v62, 256, 1);
                  else
LABEL_143:
                    v65 = (*(_QWORD *)(*(_QWORD *)(v62 + 16) + 24LL) >> 8) & 1LL;
                  v45 = ((_BYTE)v65 == 0) + v95;
                }
              }
LABEL_65:
              if ( a3 <= v45 || sub_2EE6AD0(v38, v86, v85) && sub_2EE6AD0(v36, v86, v85) && v45 > 1 && (v93 || v91) )
                goto LABEL_100;
              goto LABEL_67;
            }
          }
        }
        else if ( !sub_2E322F0(v38, v36) || !v93 )
        {
          goto LABEL_57;
        }
LABEL_100:
        if ( v95 <= v81 )
        {
          if ( (_DWORD *)v79 != v78 || v95 != v81 )
            goto LABEL_67;
          v57 = (__m128i *)v6[14];
          v56 = (const __m128i *)v6[15];
        }
        else
        {
          v55 = (__m128i *)v6[13];
          if ( v55 != (__m128i *)v6[14] )
            v6[14] = v55;
          v56 = (const __m128i *)v6[15];
          v96.m128i_i64[0] = (__int64)v78;
          v96.m128i_i64[1] = v83;
          if ( v55 == v56 )
          {
            sub_34C2E70(v76, v56, &v96);
            v57 = (__m128i *)v6[14];
            v56 = (const __m128i *)v6[15];
          }
          else
          {
            if ( v55 )
            {
              *v55 = _mm_loadu_si128(&v96);
              v55 = (__m128i *)v6[14];
              v56 = (const __m128i *)v6[15];
            }
            v57 = v55 + 1;
            v6[14] = v57;
          }
          v79 = (__int64)v78;
        }
        v96.m128i_i64[0] = v7;
        v96.m128i_i64[1] = v80;
        if ( v56 == v57 )
        {
          sub_34C2E70(v76, v56, &v96);
        }
        else
        {
          if ( v57 )
          {
            *v57 = _mm_loadu_si128(&v96);
            v57 = (__m128i *)v6[14];
          }
          v6[14] = v57 + 1;
        }
        v81 = v95;
LABEL_67:
        if ( v84 == (_DWORD *)v7 )
          break;
        v7 -= 24;
      }
      while ( a2 == *(_DWORD *)v7 );
LABEL_69:
      if ( v84 == v87 )
        return v81;
      v78 -= 6;
    }
  }
  return 0;
}
