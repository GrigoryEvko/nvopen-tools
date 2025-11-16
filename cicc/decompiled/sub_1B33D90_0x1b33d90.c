// Function: sub_1B33D90
// Address: 0x1b33d90
//
__int64 __fastcall sub_1B33D90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned int v6; // r15d
  __int64 v7; // r13
  __int64 *v8; // rax
  __int64 *v10; // rdx
  __int64 v11; // rsi
  char v12; // dl
  __int64 *v13; // rcx
  __int64 v14; // rax
  int v15; // eax
  unsigned __int64 v16; // rdi
  __int64 *v17; // rax
  _QWORD *v19; // rbx
  int v20; // r8d
  int v21; // r9d
  _QWORD *v22; // r15
  unsigned __int8 v23; // al
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rbx
  char v27; // dl
  __int64 *v28; // rax
  __int64 *v29; // rsi
  __int64 *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdi
  unsigned int v33; // eax
  __int64 v34; // r9
  unsigned int v35; // r12d
  __int64 *v36; // rax
  _QWORD *v37; // rcx
  __int64 v38; // r13
  __int64 v39; // r12
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  int v43; // r13d
  __int64 v44; // rax
  __int64 v45; // rdx
  int v46; // edx
  __int64 v47; // r13
  __int64 v48; // rcx
  __int64 v49; // r14
  int v50; // r10d
  int v51; // r10d
  __int64 v52; // r11
  unsigned int v53; // ecx
  int v54; // edx
  __int64 v55; // r9
  __int64 *v56; // rdi
  int v57; // edx
  int v58; // edi
  int v59; // r9d
  int v60; // r9d
  __int64 v61; // r10
  int v62; // ecx
  unsigned int v63; // ebx
  __int64 *v64; // rdi
  int v65; // edi
  __int64 v66; // rbx
  char v67; // dl
  __int64 *v68; // rax
  __int64 *v69; // rsi
  __int64 *v70; // rcx
  __int64 v71; // rax
  __int64 v72; // [rsp+8h] [rbp-2B8h]
  int v73; // [rsp+10h] [rbp-2B0h]
  __int64 v74; // [rsp+10h] [rbp-2B0h]
  unsigned __int8 v75; // [rsp+2Fh] [rbp-291h]
  __int64 v77; // [rsp+48h] [rbp-278h] BYREF
  _BYTE *v78; // [rsp+50h] [rbp-270h] BYREF
  __int64 v79; // [rsp+58h] [rbp-268h]
  _BYTE v80[256]; // [rsp+60h] [rbp-260h] BYREF
  __int64 v81; // [rsp+160h] [rbp-160h] BYREF
  __int64 *v82; // [rsp+168h] [rbp-158h]
  __int64 *v83; // [rsp+170h] [rbp-150h]
  __int64 v84; // [rsp+178h] [rbp-148h]
  int v85; // [rsp+180h] [rbp-140h]
  _BYTE v86[312]; // [rsp+188h] [rbp-138h] BYREF

  v6 = 0;
  v7 = *(_QWORD *)(a1 + 8);
  v79 = 0x2000000000LL;
  v8 = (__int64 *)v86;
  v78 = v80;
  v81 = 0;
  v82 = (__int64 *)v86;
  v83 = (__int64 *)v86;
  v84 = 32;
  v85 = 0;
  if ( !v7 )
    goto LABEL_21;
  v10 = (__int64 *)v86;
  while ( 1 )
  {
    if ( v10 == v8 )
    {
      v11 = (__int64)&v8[HIDWORD(v84)];
      if ( (__int64 *)v11 != v8 )
      {
        v13 = 0;
        while ( *v8 != v7 )
        {
          if ( *v8 == -2 )
            v13 = v8;
          if ( (__int64 *)v11 == ++v8 )
          {
            if ( !v13 )
              goto LABEL_24;
            *v13 = v7;
            --v85;
            ++v81;
            goto LABEL_15;
          }
        }
        goto LABEL_4;
      }
LABEL_24:
      if ( HIDWORD(v84) < (unsigned int)v84 )
        break;
    }
    v11 = v7;
    sub_16CCBA0((__int64)&v81, v7);
    if ( v12 )
    {
LABEL_15:
      v14 = (unsigned int)v79;
      if ( (unsigned int)v79 < HIDWORD(v79) )
        goto LABEL_16;
      goto LABEL_26;
    }
LABEL_4:
    v7 = *(_QWORD *)(v7 + 8);
    if ( !v7 )
      goto LABEL_17;
LABEL_5:
    v10 = v83;
    v8 = v82;
  }
  ++HIDWORD(v84);
  *(_QWORD *)v11 = v7;
  v14 = (unsigned int)v79;
  ++v81;
  if ( (unsigned int)v79 < HIDWORD(v79) )
    goto LABEL_16;
LABEL_26:
  v11 = (__int64)v80;
  sub_16CD150((__int64)&v78, v80, 0, 8, a5, a6);
  v14 = (unsigned int)v79;
LABEL_16:
  *(_QWORD *)&v78[8 * v14] = v7;
  LODWORD(v79) = v79 + 1;
  v7 = *(_QWORD *)(v7 + 8);
  if ( v7 )
    goto LABEL_5;
LABEL_17:
  v15 = v79;
  if ( !(_DWORD)v79 )
    goto LABEL_18;
  v75 = 0;
  while ( 2 )
  {
    v19 = *(_QWORD **)&v78[8 * v15 - 8];
    LODWORD(v79) = v15 - 1;
    v22 = sub_1648700((__int64)v19);
    v23 = *((_BYTE *)v22 + 16);
    if ( v23 <= 0x17u )
    {
LABEL_18:
      v16 = (unsigned __int64)v83;
      v17 = v82;
      v6 = 0;
      goto LABEL_19;
    }
    if ( v23 != 55 )
    {
      if ( v23 != 78 )
      {
        if ( v23 != 71 )
        {
          if ( v23 == 56 )
          {
            if ( *v19 != v22[-3 * (*((_DWORD *)v22 + 5) & 0xFFFFFFF)] )
              goto LABEL_18;
            v26 = v22[1];
            if ( !v26 )
              goto LABEL_56;
            while ( 1 )
            {
              v28 = v82;
              if ( v83 == v82 )
              {
                v29 = &v82[HIDWORD(v84)];
                if ( v82 != v29 )
                {
                  v30 = 0;
                  while ( v26 != *v28 )
                  {
                    if ( *v28 == -2 )
                      v30 = v28;
                    if ( v29 == ++v28 )
                    {
                      if ( !v30 )
                        goto LABEL_90;
                      *v30 = v26;
                      --v85;
                      ++v81;
                      goto LABEL_51;
                    }
                  }
                  goto LABEL_41;
                }
LABEL_90:
                if ( HIDWORD(v84) < (unsigned int)v84 )
                  break;
              }
              sub_16CCBA0((__int64)&v81, v26);
              if ( v27 )
                goto LABEL_51;
LABEL_41:
              v26 = *(_QWORD *)(v26 + 8);
              if ( !v26 )
                goto LABEL_56;
            }
            ++HIDWORD(v84);
            *v29 = v26;
            ++v81;
LABEL_51:
            v31 = (unsigned int)v79;
            if ( (unsigned int)v79 >= HIDWORD(v79) )
            {
              sub_16CD150((__int64)&v78, v80, 0, 8, v20, v21);
              v31 = (unsigned int)v79;
            }
            *(_QWORD *)&v78[8 * v31] = v26;
            LODWORD(v79) = v79 + 1;
            goto LABEL_41;
          }
          if ( v23 != 54 )
            goto LABEL_18;
LABEL_56:
          v11 = *(unsigned int *)(a2 + 24);
          if ( (_DWORD)v11 )
          {
            v34 = *(_QWORD *)(a2 + 8);
            v35 = (v11 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
            v36 = (__int64 *)(v34 + 8LL * v35);
            v37 = (_QWORD *)*v36;
            if ( v22 == (_QWORD *)*v36 )
            {
LABEL_58:
              v15 = v79;
              if ( (_DWORD)v79 )
                continue;
              v6 = v75;
              v16 = (unsigned __int64)v83;
              v17 = v82;
              goto LABEL_19;
            }
            v56 = 0;
            v57 = 1;
            while ( v37 != (_QWORD *)-8LL )
            {
              if ( v37 == (_QWORD *)-16LL && !v56 )
                v56 = v36;
              v35 = (v11 - 1) & (v57 + v35);
              v36 = (__int64 *)(v34 + 8LL * v35);
              v37 = (_QWORD *)*v36;
              if ( v22 == (_QWORD *)*v36 )
                goto LABEL_58;
              ++v57;
            }
            if ( v56 )
              v36 = v56;
            v58 = *(_DWORD *)(a2 + 16);
            ++*(_QWORD *)a2;
            v54 = v58 + 1;
            if ( 4 * (v58 + 1) < (unsigned int)(3 * v11) )
            {
              if ( (int)v11 - *(_DWORD *)(a2 + 20) - v54 <= (unsigned int)v11 >> 3 )
              {
                sub_163D1D0(a2, v11);
                v59 = *(_DWORD *)(a2 + 24);
                if ( !v59 )
                {
LABEL_148:
                  ++*(_DWORD *)(a2 + 16);
                  BUG();
                }
                v60 = v59 - 1;
                v61 = *(_QWORD *)(a2 + 8);
                v62 = 1;
                v63 = v60 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
                v54 = *(_DWORD *)(a2 + 16) + 1;
                v64 = 0;
                v36 = (__int64 *)(v61 + 8LL * v63);
                v11 = *v36;
                if ( v22 != (_QWORD *)*v36 )
                {
                  while ( v11 != -8 )
                  {
                    if ( !v64 && v11 == -16 )
                      v64 = v36;
                    v63 = v60 & (v62 + v63);
                    v36 = (__int64 *)(v61 + 8LL * v63);
                    v11 = *v36;
                    if ( v22 == (_QWORD *)*v36 )
                      goto LABEL_87;
                    ++v62;
                  }
                  if ( v64 )
                    v36 = v64;
                }
              }
              goto LABEL_87;
            }
          }
          else
          {
            ++*(_QWORD *)a2;
          }
          v11 = (unsigned int)(2 * v11);
          sub_163D1D0(a2, v11);
          v50 = *(_DWORD *)(a2 + 24);
          if ( !v50 )
            goto LABEL_148;
          v51 = v50 - 1;
          v52 = *(_QWORD *)(a2 + 8);
          v53 = v51 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
          v54 = *(_DWORD *)(a2 + 16) + 1;
          v36 = (__int64 *)(v52 + 8LL * v53);
          v55 = *v36;
          if ( v22 != (_QWORD *)*v36 )
          {
            v11 = 0;
            v65 = 1;
            while ( v55 != -8 )
            {
              if ( !v11 && v55 == -16 )
                v11 = (__int64)v36;
              v53 = v51 & (v65 + v53);
              v36 = (__int64 *)(v52 + 8LL * v53);
              v55 = *v36;
              if ( v22 == (_QWORD *)*v36 )
                goto LABEL_87;
              ++v65;
            }
            if ( v11 )
              v36 = (__int64 *)v11;
          }
LABEL_87:
          *(_DWORD *)(a2 + 16) = v54;
          if ( *v36 != -8 )
            --*(_DWORD *)(a2 + 20);
          *v36 = (__int64)v22;
          goto LABEL_58;
        }
        v66 = v22[1];
        if ( !v66 )
          goto LABEL_56;
        while ( 1 )
        {
          v68 = v82;
          if ( v83 == v82 )
          {
            v69 = &v82[HIDWORD(v84)];
            if ( v82 != v69 )
            {
              v70 = 0;
              while ( v66 != *v68 )
              {
                if ( *v68 == -2 )
                  v70 = v68;
                if ( v69 == ++v68 )
                {
                  if ( !v70 )
                    goto LABEL_144;
                  *v70 = v66;
                  --v85;
                  ++v81;
                  goto LABEL_141;
                }
              }
              goto LABEL_131;
            }
LABEL_144:
            if ( HIDWORD(v84) < (unsigned int)v84 )
              break;
          }
          sub_16CCBA0((__int64)&v81, v66);
          if ( v67 )
            goto LABEL_141;
LABEL_131:
          v66 = *(_QWORD *)(v66 + 8);
          if ( !v66 )
            goto LABEL_56;
        }
        ++HIDWORD(v84);
        *v69 = v66;
        ++v81;
LABEL_141:
        v71 = (unsigned int)v79;
        if ( (unsigned int)v79 >= HIDWORD(v79) )
        {
          sub_16CD150((__int64)&v78, v80, 0, 8, v20, v21);
          v71 = (unsigned int)v79;
        }
        *(_QWORD *)&v78[8 * v71] = v66;
        LODWORD(v79) = v79 + 1;
        goto LABEL_131;
      }
      v38 = *(v22 - 3);
      if ( *(_BYTE *)(v38 + 16) )
        goto LABEL_18;
      if ( (*(_BYTE *)(v38 + 33) & 0x20) != 0 )
      {
        if ( (unsigned int)(*(_DWORD *)(v38 + 36) - 116) > 1 )
          goto LABEL_18;
        goto LABEL_56;
      }
      if ( (*(_BYTE *)(v38 + 18) & 1) != 0 )
      {
        sub_15E08E0(*(v22 - 3), v11);
        v39 = *(_QWORD *)(v38 + 88);
        if ( (*(_BYTE *)(v38 + 18) & 1) != 0 )
          sub_15E08E0(v38, v11);
        v40 = *(_QWORD *)(v38 + 88);
      }
      else
      {
        v39 = *(_QWORD *)(v38 + 88);
        v40 = v39;
      }
      v72 = v40 + 40LL * *(_QWORD *)(v38 + 96);
      v77 = *(_QWORD *)(v38 + 112);
      v73 = *((_DWORD *)v22 + 5) & 0xFFFFFFF;
      if ( *((char *)v22 + 23) < 0 )
      {
        v41 = sub_1648A40((__int64)v22);
        if ( *((char *)v22 + 23) >= 0 )
        {
          if ( (unsigned int)((v41 + v42) >> 4) )
LABEL_150:
            BUG();
        }
        else if ( (unsigned int)((v41 + v42 - sub_1648A40((__int64)v22)) >> 4) )
        {
          if ( *((char *)v22 + 23) >= 0 )
            goto LABEL_150;
          v43 = *(_DWORD *)(sub_1648A40((__int64)v22) + 8);
          if ( *((char *)v22 + 23) >= 0 )
            BUG();
          v44 = sub_1648A40((__int64)v22);
          v46 = *(_DWORD *)(v44 + v45 - 4) - v43;
          goto LABEL_70;
        }
      }
      v46 = 0;
LABEL_70:
      v47 = 0;
      v48 = (unsigned int)(v73 - 1 - v46);
      if ( v73 - 1 != v46 )
      {
        v74 = a2;
        v49 = v48;
        do
        {
          if ( *v19 == v22[3 * (v47 - (*((_DWORD *)v22 + 5) & 0xFFFFFFF))]
            && (v72 == v39 || !(unsigned __int8)sub_1560290(&v77, v47, 6)) )
          {
            goto LABEL_18;
          }
          ++v47;
          v39 += 40;
        }
        while ( v47 != v49 );
        a2 = v74;
      }
      goto LABEL_56;
    }
    break;
  }
  v24 = *(v22 - 3);
  if ( !v24 )
    goto LABEL_18;
  if ( a1 != v24 )
    goto LABEL_18;
  v25 = *(v22 - 6);
  if ( *(_BYTE *)(v25 + 16) != 54 )
    goto LABEL_18;
  v32 = *(_QWORD *)(v25 - 24);
  if ( *(_BYTE *)(v32 + 16) != 17 )
    goto LABEL_18;
  v33 = sub_15E0450(v32);
  v75 = v33;
  if ( (_BYTE)v33 )
    goto LABEL_56;
  v6 = v33;
  v16 = (unsigned __int64)v83;
  v17 = v82;
LABEL_19:
  if ( (__int64 *)v16 != v17 )
    _libc_free(v16);
LABEL_21:
  if ( v78 != v80 )
    _libc_free((unsigned __int64)v78);
  return v6;
}
