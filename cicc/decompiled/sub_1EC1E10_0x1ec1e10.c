// Function: sub_1EC1E10
// Address: 0x1ec1e10
//
__int64 __fastcall sub_1EC1E10(_QWORD *a1, unsigned int a2, __int64 a3, __int64 a4, _QWORD *a5, unsigned __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r12
  unsigned int v10; // ebx
  __int64 v11; // rcx
  _QWORD *v12; // r13
  unsigned int v13; // edx
  unsigned __int64 v14; // rcx
  unsigned __int16 v15; // ax
  __int16 *v16; // rdx
  __int16 v17; // ax
  __int16 *v18; // r13
  _QWORD *v19; // rax
  _QWORD *v20; // r15
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r10
  __int64 v25; // rsi
  __int64 *v26; // rdx
  __int64 *v27; // rdi
  __int64 v28; // rdx
  unsigned __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  unsigned int v33; // r14d
  float v34; // xmm1_4
  __int64 v35; // r10
  __int64 v36; // r11
  __int64 v37; // rdx
  int v38; // eax
  __int64 v39; // r10
  __int64 v40; // rcx
  __int64 v41; // rsi
  __int64 v42; // rcx
  unsigned int v43; // edx
  __int64 result; // rax
  _WORD *v45; // rcx
  unsigned __int16 *v46; // rdx
  unsigned __int16 v47; // r15
  unsigned __int16 *v48; // r14
  __int64 *v49; // rcx
  __int64 *v50; // r9
  __int64 v51; // r10
  __int64 v52; // rdx
  unsigned int v53; // esi
  unsigned int v54; // ecx
  __int64 v55; // rdi
  _QWORD *v56; // rax
  _QWORD *v57; // r8
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // rax
  unsigned __int64 v61; // [rsp+8h] [rbp-E8h]
  _QWORD *v62; // [rsp+10h] [rbp-E0h]
  __int64 v63; // [rsp+18h] [rbp-D8h]
  _QWORD *v64; // [rsp+20h] [rbp-D0h]
  unsigned int v65; // [rsp+28h] [rbp-C8h]
  _QWORD *v66; // [rsp+30h] [rbp-C0h]
  _QWORD *v67; // [rsp+30h] [rbp-C0h]
  signed __int64 v69; // [rsp+40h] [rbp-B0h]
  signed __int64 v70; // [rsp+48h] [rbp-A8h]
  unsigned __int16 v71; // [rsp+50h] [rbp-A0h]
  __int64 *v72; // [rsp+50h] [rbp-A0h]
  char v73; // [rsp+50h] [rbp-A0h]
  __int64 *v74; // [rsp+50h] [rbp-A0h]
  _QWORD *v75; // [rsp+50h] [rbp-A0h]
  __int16 v76; // [rsp+58h] [rbp-98h]
  unsigned int v77; // [rsp+5Ch] [rbp-94h]
  __int64 v78; // [rsp+60h] [rbp-90h] BYREF
  __int64 *v79; // [rsp+68h] [rbp-88h] BYREF
  __int64 v80; // [rsp+70h] [rbp-80h]
  __int64 v81; // [rsp+78h] [rbp-78h] BYREF
  unsigned __int64 v82; // [rsp+80h] [rbp-70h]

  v7 = a1[123];
  v76 = a2;
  v8 = *(_QWORD *)(v7 + 280);
  v9 = *(_QWORD *)(v7 + 200);
  v10 = *(_DWORD *)(v7 + 208) - 1;
  v70 = *(_QWORD *)(v8 + 8);
  if ( *(_BYTE *)(v8 + 32) )
    v70 = *(_QWORD *)(v8 + 8) & 0xFFFFFFFFFFFFFFF8LL;
  v69 = *(_QWORD *)(v8 + 16);
  if ( *(_BYTE *)(v8 + 33) )
    v69 = *(_QWORD *)(v8 + 16) & 0xFFFFFFFFFFFFFFF8LL | 6;
  *(_DWORD *)(a3 + 8) = 0;
  if ( *(_DWORD *)(a3 + 12) < v10 )
    sub_16CD150(a3, (const void *)(a3 + 16), v10, 4, (int)a5, a6);
  *(_DWORD *)(a3 + 8) = v10;
  if ( 4LL * v10 )
    memset(*(void **)a3, 0, 4LL * v10);
  v11 = a1[87];
  if ( !v11 )
    BUG();
  v64 = (_QWORD *)a3;
  v12 = a1;
  v63 = 24LL * a2;
  v13 = *(_DWORD *)(*(_QWORD *)(v11 + 8) + v63 + 16);
  v14 = *(_QWORD *)(v11 + 56) + 2LL * (v13 >> 4);
  v15 = *(_WORD *)v14 + a2 * (v13 & 0xF);
  v16 = (__int16 *)(v14 + 2);
  v71 = v15;
  v65 = (v70 >> 1) & 3;
  v77 = (v69 >> 1) & 3;
LABEL_13:
  v19 = v12;
  v18 = v16;
  v20 = v19;
  while ( v18 )
  {
    v21 = sub_2103840(v20[34], *(_QWORD *)(v20[123] + 40LL), v71, v14, a5, a6);
    if ( (unsigned int)sub_20FD0B0(v21, 1) )
    {
      v22 = *(_QWORD *)(v20[34] + 384LL) + 216LL * v71;
      v23 = *(unsigned int *)(v22 + 200);
      v24 = v22 + 8;
      v25 = *(unsigned int *)(v22 + 204);
      v79 = &v81;
      v78 = v22 + 8;
      v80 = 0x400000000LL;
      if ( !(_DWORD)v23 )
      {
        if ( (_DWORD)v25 )
        {
          v26 = (__int64 *)(v22 + 16);
          do
          {
            if ( (*(_DWORD *)((*v26 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v26 >> 1) & 3) > (*(_DWORD *)((v70 & 0xFFFFFFFFFFFFFFF8LL) + 24) | v65) )
              break;
            v23 = (unsigned int)(v23 + 1);
            v26 += 2;
          }
          while ( (_DWORD)v23 != (_DWORD)v25 );
        }
        v81 = v24;
        v27 = &v81;
        v28 = 1;
        v14 = v25 | (v23 << 32);
        LODWORD(v80) = 1;
        v82 = v14;
        v29 = HIDWORD(v14);
        goto LABEL_22;
      }
      v39 = v22 + 16;
      if ( (_DWORD)v25 )
      {
        v22 += 104;
        v40 = 0;
        do
        {
          a6 = *(_QWORD *)v22 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_DWORD *)(a6 + 24) | (unsigned int)(*(__int64 *)v22 >> 1) & 3) > (*(_DWORD *)((v70
                                                                                               & 0xFFFFFFFFFFFFFFF8LL)
                                                                                              + 24)
                                                                                  | v65) )
            break;
          v40 = (unsigned int)(v40 + 1);
          v22 += 8;
        }
        while ( (_DWORD)v40 != (_DWORD)v25 );
      }
      else
      {
        v40 = 0;
      }
      v81 = v39;
      v14 = v25 | (v40 << 32);
      LODWORD(v80) = 1;
      v82 = v14;
      if ( HIDWORD(v14) >= (unsigned int)v14 )
        goto LABEL_11;
      sub_1EC1C10((__int64)&v78, v70, v22, v14, (unsigned __int64)a5);
      v28 = (unsigned int)v80;
      v27 = v79;
      if ( (_DWORD)v80 )
      {
        LODWORD(v29) = *((_DWORD *)v79 + 3);
        v14 = *((unsigned int *)v79 + 2);
LABEL_22:
        a5 = v64;
        LODWORD(v30) = 0;
        a6 = v69 & 0xFFFFFFFFFFFFFFF8LL;
        while ( (unsigned int)v29 < (unsigned int)v14 )
        {
          v31 = (__int64)&v27[2 * v28 - 2];
          v32 = *(_QWORD *)(*(_QWORD *)v31 + 16LL * *(unsigned int *)(v31 + 12));
          v14 = *(_DWORD *)((v32 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v32 >> 1) & 3;
          if ( (unsigned int)v14 >= (*(_DWORD *)(a6 + 24) | v77) )
            break;
          while ( 1 )
          {
            v33 = v30;
            v30 = (unsigned int)(v30 + 1);
            if ( (unsigned int)v14 <= (*(_DWORD *)((*(_QWORD *)(v9 + 8 * v30) & 0xFFFFFFFFFFFFFFF8LL) + 24) | 3u) )
              break;
            if ( (_DWORD)v30 == v10 )
              goto LABEL_34;
          }
          if ( v33 == v10 )
            break;
          v34 = *(float *)(*(_QWORD *)(*(_QWORD *)v31 + 8LL * *(unsigned int *)(v31 + 12) + 128) + 116LL);
          while ( 1 )
          {
            *(float *)(*a5 + 4LL * v33) = fmaxf(v34, *(float *)(*a5 + 4LL * v33));
            v35 = v78;
            v14 = v33 + 1;
            v36 = *(_QWORD *)(v79[2 * (unsigned int)v80 - 2] + 16LL * HIDWORD(v79[2 * (unsigned int)v80 - 1]) + 8);
            if ( *(_DWORD *)((*(_QWORD *)(v9 + 8 * v14) & 0xFFFFFFFFFFFFFFF8LL) + 24) >= (*(_DWORD *)((v36 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                        | (unsigned int)(v36 >> 1) & 3) )
              break;
            if ( (_DWORD)v14 == v10 )
            {
              v27 = v79;
              goto LABEL_34;
            }
            ++v33;
          }
          v37 = (__int64)&v79[2 * (unsigned int)v80 - 2];
          v38 = *(_DWORD *)(v37 + 12) + 1;
          *(_DWORD *)(v37 + 12) = v38;
          v28 = (unsigned int)v80;
          v27 = v79;
          v14 = 16LL * (unsigned int)v80;
          if ( v38 == *(_DWORD *)((char *)v79 + v14 - 8) )
          {
            v41 = *(unsigned int *)(v35 + 192);
            if ( (_DWORD)v41 )
            {
              v61 = a6;
              v62 = a5;
              sub_39460A0(&v79, v41);
              v28 = (unsigned int)v80;
              v27 = v79;
              a5 = v62;
              a6 = v61;
            }
          }
          if ( !(_DWORD)v28 )
            break;
          LODWORD(v29) = *((_DWORD *)v27 + 3);
          v14 = *((unsigned int *)v27 + 2);
          LODWORD(v30) = v33;
        }
      }
LABEL_34:
      if ( v27 != &v81 )
        _libc_free((unsigned __int64)v27);
    }
LABEL_11:
    v17 = *v18;
    v16 = 0;
    ++v18;
    v71 += v17;
    if ( !v17 )
    {
      v12 = v20;
      goto LABEL_13;
    }
  }
  v42 = a1[87];
  if ( !v42 )
    BUG();
  v43 = *(_DWORD *)(*(_QWORD *)(v42 + 8) + v63 + 16);
  result = v43 & 0xF;
  v45 = (_WORD *)(*(_QWORD *)(v42 + 56) + 2LL * (v43 >> 4));
  v46 = v45 + 1;
  v47 = *v45 + v76 * result;
  while ( 1 )
  {
    v48 = v46;
    if ( !v46 )
      return result;
    while ( 1 )
    {
      v49 = *(__int64 **)(*(_QWORD *)(a1[33] + 672LL) + 8LL * v47);
      if ( !v49 )
      {
        v66 = (_QWORD *)a1[33];
        v73 = qword_4FC4440[20];
        v56 = (_QWORD *)sub_22077B0(104);
        v57 = v66;
        v58 = v47;
        v59 = (__int64)v56;
        if ( v56 )
        {
          *v56 = v56 + 2;
          v56[1] = 0x200000000LL;
          v56[8] = v56 + 10;
          v56[9] = 0x200000000LL;
          if ( v73 )
          {
            v67 = v56;
            v75 = v57;
            v60 = sub_22077B0(48);
            v57 = v75;
            v59 = (__int64)v67;
            v58 = v47;
            if ( v60 )
            {
              *(_DWORD *)(v60 + 8) = 0;
              *(_QWORD *)(v60 + 16) = 0;
              *(_QWORD *)(v60 + 24) = v60 + 8;
              *(_QWORD *)(v60 + 32) = v60 + 8;
              *(_QWORD *)(v60 + 40) = 0;
            }
            v67[12] = v60;
          }
          else
          {
            v56[12] = 0;
          }
        }
        v74 = (__int64 *)v59;
        *(_QWORD *)(v57[84] + 8 * v58) = v59;
        sub_1DBA8F0(v57, v59, v47);
        v49 = v74;
      }
      v72 = v49;
      v50 = (__int64 *)sub_1DB3C70(v49, v70);
      v51 = *v72 + 24LL * *((unsigned int *)v72 + 2);
      if ( (__int64 *)v51 != v50 )
      {
        LODWORD(v52) = 0;
        do
        {
          v53 = *(_DWORD *)((*v50 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v50 >> 1) & 3;
          if ( v53 >= (*(_DWORD *)((v69 & 0xFFFFFFFFFFFFFFF8LL) + 24) | v77) )
            break;
          while ( 1 )
          {
            v54 = v52;
            v52 = (unsigned int)(v52 + 1);
            if ( v53 <= (*(_DWORD *)((*(_QWORD *)(v9 + 8 * v52) & 0xFFFFFFFFFFFFFFF8LL) + 24) | 3u) )
              break;
            if ( (_DWORD)v52 == v10 )
              goto LABEL_62;
          }
          if ( v54 == v10 )
            break;
          v55 = v54;
          while ( 1 )
          {
            LODWORD(v52) = v54;
            *(_DWORD *)(*v64 + 4 * v55) = unk_4530D80;
            v55 = ++v54;
            if ( *(_DWORD *)((*(_QWORD *)(v9 + 8LL * v54) & 0xFFFFFFFFFFFFFFF8LL) + 24) >= (*(_DWORD *)((v50[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                          | (unsigned int)(v50[1] >> 1)
                                                                                          & 3) )
              break;
            if ( v54 == v10 )
              goto LABEL_62;
          }
          v50 += 3;
        }
        while ( v50 != (__int64 *)v51 );
      }
LABEL_62:
      result = *v48;
      v46 = 0;
      ++v48;
      v47 += result;
      if ( !(_WORD)result )
        break;
      if ( !v48 )
        return result;
    }
  }
}
