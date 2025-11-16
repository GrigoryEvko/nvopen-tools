// Function: sub_2A6F200
// Address: 0x2a6f200
//
void __fastcall sub_2A6F200(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r15
  __int64 v3; // rbx
  __int64 k; // rax
  _QWORD *v5; // rdx
  unsigned int v6; // esi
  __int64 v7; // r12
  __int64 v8; // r9
  unsigned int v9; // r13d
  __int64 v10; // r8
  _QWORD *v11; // rdi
  __int64 v12; // rcx
  _QWORD *v13; // r11
  int v14; // eax
  int v15; // edx
  __int64 v16; // rcx
  __int64 v17; // rsi
  _QWORD *v18; // rax
  _QWORD *v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // r15
  __int64 *v23; // rax
  __int64 v24; // r8
  __int64 v25; // r12
  unsigned __int8 *v26; // r12
  __int64 v27; // rax
  int v28; // r14d
  __int64 v29; // rsi
  __int64 v30; // r15
  int i; // ebx
  __int64 v32; // rdx
  __int64 v33; // rcx
  int v34; // r10d
  unsigned int j; // eax
  __int64 v36; // r13
  unsigned int v37; // eax
  unsigned __int8 v38; // al
  __int64 v39; // r12
  _BYTE *v40; // r13
  __int64 v41; // rdx
  __int64 v42; // r8
  __int64 v43; // r9
  unsigned int v44; // ecx
  __int64 v45; // rsi
  __int64 v46; // rdi
  _QWORD *v47; // r12
  _QWORD *m; // r14
  _BYTE *v49; // r13
  __int64 v50; // rdx
  __int64 v51; // rsi
  unsigned int v52; // ecx
  unsigned __int8 *v53; // rax
  __int64 v54; // r8
  __int64 v55; // r13
  int v56; // eax
  __int64 v57; // rax
  __int32 v58; // ebx
  __int64 v59; // r12
  unsigned __int8 *v60; // r15
  int v61; // eax
  __int64 v62; // rdi
  unsigned int v63; // eax
  __int64 v64; // rsi
  int v65; // eax
  __int64 v66; // rdi
  unsigned int v67; // eax
  __int64 v68; // rsi
  int v69; // esi
  int v70; // r10d
  int v71; // r9d
  int v72; // eax
  int v73; // r9d
  __int64 v74; // [rsp+8h] [rbp-2A8h]
  int v75; // [rsp+10h] [rbp-2A0h]
  __int64 v76; // [rsp+18h] [rbp-298h]
  int v77; // [rsp+28h] [rbp-288h]
  unsigned __int64 v78; // [rsp+28h] [rbp-288h]
  __int64 v79; // [rsp+28h] [rbp-288h]
  __m128i v80; // [rsp+30h] [rbp-280h] BYREF
  __int64 v81; // [rsp+40h] [rbp-270h] BYREF
  __int64 v82; // [rsp+48h] [rbp-268h]
  int v83; // [rsp+50h] [rbp-260h]
  __int64 v84; // [rsp+58h] [rbp-258h]
  int v85; // [rsp+60h] [rbp-250h]
  _QWORD *v86; // [rsp+70h] [rbp-240h] BYREF
  __int64 v87; // [rsp+78h] [rbp-238h]
  _QWORD v88[70]; // [rsp+80h] [rbp-230h] BYREF

  v2 = v88;
  v3 = a1;
  v88[0] = a2;
  v86 = v88;
  v76 = a1 + 328;
  v87 = 0x4000000001LL;
  LODWORD(k) = 1;
  while ( 1 )
  {
    v5 = &v2[(unsigned int)k];
    if ( !(_DWORD)k )
      break;
    while ( 1 )
    {
      v6 = *(_DWORD *)(v3 + 352);
      LODWORD(k) = k - 1;
      v7 = *(v5 - 1);
      LODWORD(v87) = k;
      if ( !v6 )
      {
        ++*(_QWORD *)(v3 + 328);
        goto LABEL_78;
      }
      v8 = *(_QWORD *)(v3 + 336);
      v9 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
      v10 = (v6 - 1) & v9;
      v11 = (_QWORD *)(v8 + 8 * v10);
      v12 = *v11;
      if ( v7 != *v11 )
        break;
LABEL_5:
      --v5;
      if ( !(_DWORD)k )
        goto LABEL_6;
    }
    v77 = 1;
    v13 = 0;
    while ( v12 != -4096 )
    {
      if ( v12 != -8192 || v13 )
        v11 = v13;
      v10 = (v6 - 1) & (v77 + (_DWORD)v10);
      v12 = *(_QWORD *)(v8 + 8LL * (unsigned int)v10);
      if ( v7 == v12 )
        goto LABEL_5;
      ++v77;
      v13 = v11;
      v11 = (_QWORD *)(v8 + 8LL * (unsigned int)v10);
    }
    v14 = *(_DWORD *)(v3 + 344);
    if ( !v13 )
      v13 = v11;
    ++*(_QWORD *)(v3 + 328);
    v15 = v14 + 1;
    if ( 4 * (v14 + 1) < 3 * v6 )
    {
      v16 = v6 - (v15 + *(_DWORD *)(v3 + 348));
      if ( (unsigned int)v16 > v6 >> 3 )
        goto LABEL_15;
      sub_CE2A30(v76, v6);
      v65 = *(_DWORD *)(v3 + 352);
      if ( v65 )
      {
        v16 = (unsigned int)(v65 - 1);
        v66 = *(_QWORD *)(v3 + 336);
        v10 = 0;
        v67 = v16 & v9;
        v8 = 1;
        v13 = (_QWORD *)(v66 + 8LL * ((unsigned int)v16 & v9));
        v68 = *v13;
        v15 = *(_DWORD *)(v3 + 344) + 1;
        if ( v7 != *v13 )
        {
          while ( v68 != -4096 )
          {
            if ( !v10 && v68 == -8192 )
              v10 = (__int64)v13;
            v67 = v16 & (v8 + v67);
            v13 = (_QWORD *)(v66 + 8LL * v67);
            v68 = *v13;
            if ( v7 == *v13 )
              goto LABEL_15;
            v8 = (unsigned int)(v8 + 1);
          }
          goto LABEL_82;
        }
        goto LABEL_15;
      }
LABEL_115:
      ++*(_DWORD *)(v3 + 344);
      BUG();
    }
LABEL_78:
    sub_CE2A30(v76, 2 * v6);
    v61 = *(_DWORD *)(v3 + 352);
    if ( !v61 )
      goto LABEL_115;
    v16 = (unsigned int)(v61 - 1);
    v62 = *(_QWORD *)(v3 + 336);
    v63 = v16 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v13 = (_QWORD *)(v62 + 8LL * v63);
    v64 = *v13;
    v15 = *(_DWORD *)(v3 + 344) + 1;
    if ( v7 != *v13 )
    {
      v8 = 1;
      v10 = 0;
      while ( v64 != -4096 )
      {
        if ( !v10 && v64 == -8192 )
          v10 = (__int64)v13;
        v63 = v16 & (v8 + v63);
        v13 = (_QWORD *)(v62 + 8LL * v63);
        v64 = *v13;
        if ( v7 == *v13 )
          goto LABEL_15;
        v8 = (unsigned int)(v8 + 1);
      }
LABEL_82:
      if ( v10 )
        v13 = (_QWORD *)v10;
    }
LABEL_15:
    *(_DWORD *)(v3 + 344) = v15;
    if ( *v13 != -4096 )
      --*(_DWORD *)(v3 + 348);
    *v13 = v7;
    v17 = *(_QWORD *)(v7 + 40);
    if ( *(_BYTE *)(v3 + 68) )
    {
      v18 = *(_QWORD **)(v3 + 48);
      v19 = &v18[*(unsigned int *)(v3 + 60)];
      if ( v18 != v19 )
      {
        while ( v17 != *v18 )
        {
          if ( v19 == ++v18 )
            goto LABEL_31;
        }
        goto LABEL_22;
      }
LABEL_31:
      v2 = v86;
      LODWORD(k) = v87;
    }
    else
    {
      if ( !sub_C8CA60(v3 + 40, v17) )
        goto LABEL_31;
LABEL_22:
      if ( *(_BYTE *)v7 == 30 )
      {
        v20 = *(unsigned int *)(v3 + 256);
        v21 = *(_QWORD *)(v3 + 240);
        v22 = *(_QWORD *)(*(_QWORD *)(v7 + 40) + 72LL);
        if ( (_DWORD)v20 )
        {
          v16 = ((_DWORD)v20 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
          v23 = (__int64 *)(v21 + 16 * v16);
          v24 = *v23;
          if ( v22 == *v23 )
          {
LABEL_25:
            v20 = v21 + 16 * v20;
            if ( v23 != (__int64 *)v20 )
            {
              v25 = *(_QWORD *)(v3 + 264) + 48LL * *((unsigned int *)v23 + 2);
              if ( v25 != *(_QWORD *)(v3 + 264) + 48LL * *(unsigned int *)(v3 + 272) )
              {
                v26 = (unsigned __int8 *)(v25 + 8);
                v81 = 0;
                sub_22C0090(v26);
                sub_22C0650((__int64)v26, (unsigned __int8 *)&v81);
                sub_22C0090((unsigned __int8 *)&v81);
                if ( !v22 )
                  goto LABEL_28;
                goto LABEL_48;
              }
            }
          }
          else
          {
            v56 = 1;
            while ( v24 != -4096 )
            {
              v71 = v56 + 1;
              v16 = ((_DWORD)v20 - 1) & (unsigned int)(v56 + v16);
              v23 = (__int64 *)(v21 + 16LL * (unsigned int)v16);
              v24 = *v23;
              if ( v22 == *v23 )
                goto LABEL_25;
              v56 = v71;
            }
          }
        }
        if ( !(unsigned __int8)sub_B19060(v3 + 360, v22, v20, v16) )
          goto LABEL_28;
        v75 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(v22 + 24) + 16LL) + 12LL);
        if ( v75 )
        {
          v79 = v22;
          v57 = v3 + 280;
          v74 = v3;
          v58 = 0;
          v59 = v57;
          do
          {
            v80.m128i_i32[2] = v58;
            v81 = 0;
            ++v58;
            v80.m128i_i64[0] = v79;
            v60 = (unsigned __int8 *)sub_2A6EC30(v59, &v80);
            sub_22C0090(v60);
            sub_22C0650((__int64)v60, (unsigned __int8 *)&v81);
            sub_22C0090((unsigned __int8 *)&v81);
          }
          while ( v58 != v75 );
          v22 = v79;
          v3 = v74;
        }
      }
      else
      {
        v27 = *(_QWORD *)(v7 + 8);
        if ( *(_BYTE *)(v27 + 8) == 15 )
        {
          v28 = *(_DWORD *)(v27 + 12);
          if ( !v28 )
            goto LABEL_28;
          v29 = 0;
          v30 = v3;
          for ( i = 0; i != v28; ++i )
          {
            v32 = *(unsigned int *)(v30 + 192);
            v33 = *(_QWORD *)(v30 + 176);
            if ( (_DWORD)v32 )
            {
              v8 = (unsigned int)(v32 - 1);
              v34 = 1;
              v78 = (unsigned __int64)(((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4)) << 32;
              for ( j = v8
                      & (((0xBF58476D1CE4E5B9LL * (v78 | (unsigned int)(37 * i))) >> 31)
                       ^ (484763065 * (v78 | (37 * i)))); ; j = v8 & v37 )
              {
                v10 = j;
                v36 = v33 + 56LL * j;
                if ( v7 == *(_QWORD *)v36 && *(_DWORD *)(v36 + 8) == i )
                  break;
                if ( *(_QWORD *)v36 == -4096 && *(_DWORD *)(v36 + 8) == -1 )
                  goto LABEL_46;
                v37 = v34 + j;
                ++v34;
              }
              if ( v36 != v33 + 56 * v32 )
              {
                v81 = 0;
                sub_22C0090((unsigned __int8 *)(v36 + 16));
                v38 = v81;
                *(_WORD *)(v36 + 16) = (unsigned __int8)v81;
                if ( v38 > 3u )
                {
                  if ( (unsigned __int8)(v38 - 4) <= 1u )
                  {
                    *(_DWORD *)(v36 + 32) = v83;
                    *(_QWORD *)(v36 + 24) = v82;
                    v83 = 0;
                    *(_DWORD *)(v36 + 48) = v85;
                    *(_QWORD *)(v36 + 40) = v84;
                    v85 = 0;
                    *(_BYTE *)(v36 + 17) = BYTE1(v81);
                  }
                }
                else if ( v38 > 1u )
                {
                  *(_QWORD *)(v36 + 24) = v82;
                }
                LOBYTE(v81) = 0;
                sub_22C0090((unsigned __int8 *)&v81);
                v29 = v7;
              }
            }
LABEL_46:
            ;
          }
          v3 = v30;
          v22 = v29;
          if ( !v29 )
            goto LABEL_28;
        }
        else
        {
          v50 = *(unsigned int *)(v3 + 160);
          v51 = *(_QWORD *)(v3 + 144);
          if ( !(_DWORD)v50 )
            goto LABEL_28;
          v52 = (v50 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
          v53 = (unsigned __int8 *)(v51 + 48LL * v52);
          v54 = *(_QWORD *)v53;
          if ( v7 != *(_QWORD *)v53 )
          {
            v72 = 1;
            while ( v54 != -4096 )
            {
              v73 = v72 + 1;
              v52 = (v50 - 1) & (v72 + v52);
              v53 = (unsigned __int8 *)(v51 + 48LL * v52);
              v54 = *(_QWORD *)v53;
              if ( v7 == *(_QWORD *)v53 )
                goto LABEL_68;
              v72 = v73;
            }
LABEL_28:
            LODWORD(k) = v87;
            goto LABEL_29;
          }
LABEL_68:
          if ( v53 == (unsigned __int8 *)(v51 + 48 * v50) )
            goto LABEL_28;
          v55 = (__int64)(v53 + 8);
          v22 = v7;
          v81 = 0;
          sub_22C0090(v53 + 8);
          sub_22C0650(v55, (unsigned __int8 *)&v81);
          sub_22C0090((unsigned __int8 *)&v81);
        }
      }
LABEL_48:
      v39 = *(_QWORD *)(v22 + 16);
      for ( k = (unsigned int)v87; v39; v39 = *(_QWORD *)(v39 + 8) )
      {
        v40 = *(_BYTE **)(v39 + 24);
        if ( *v40 > 0x1Cu )
        {
          if ( k + 1 > (unsigned __int64)HIDWORD(v87) )
          {
            sub_C8D5F0((__int64)&v86, v88, k + 1, 8u, v10, v8);
            k = (unsigned int)v87;
          }
          v86[k] = v40;
          k = (unsigned int)(v87 + 1);
          LODWORD(v87) = v87 + 1;
        }
      }
      v41 = *(unsigned int *)(v3 + 2592);
      v42 = *(_QWORD *)(v3 + 2576);
      if ( (_DWORD)v41 )
      {
        v43 = (unsigned int)(v41 - 1);
        v44 = v43 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v45 = v42 + 72LL * v44;
        v46 = *(_QWORD *)v45;
        if ( *(_QWORD *)v45 == v22 )
        {
LABEL_56:
          if ( v45 != v42 + 72 * v41 )
          {
            v47 = *(_QWORD **)(v45 + 40);
            for ( m = &v47[*(unsigned int *)(v45 + 48)]; m != v47; ++v47 )
            {
              v49 = (_BYTE *)*v47;
              if ( *(_BYTE *)*v47 > 0x1Cu )
              {
                if ( k + 1 > (unsigned __int64)HIDWORD(v87) )
                {
                  sub_C8D5F0((__int64)&v86, v88, k + 1, 8u, v42, v43);
                  k = (unsigned int)v87;
                }
                v86[k] = v49;
                k = (unsigned int)(v87 + 1);
                LODWORD(v87) = v87 + 1;
              }
            }
          }
        }
        else
        {
          v69 = 1;
          while ( v46 != -4096 )
          {
            v70 = v69 + 1;
            v44 = v43 & (v69 + v44);
            v45 = v42 + 72LL * v44;
            v46 = *(_QWORD *)v45;
            if ( *(_QWORD *)v45 == v22 )
              goto LABEL_56;
            v69 = v70;
          }
        }
      }
LABEL_29:
      v2 = v86;
    }
  }
LABEL_6:
  if ( v2 != v88 )
    _libc_free((unsigned __int64)v2);
}
