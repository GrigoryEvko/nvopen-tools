// Function: sub_2E45750
// Address: 0x2e45750
//
unsigned __int64 __fastcall sub_2E45750(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  int v7; // r13d
  __int16 *v8; // r14
  unsigned int v9; // esi
  __int64 v10; // r9
  int *v11; // rdx
  int v12; // ebx
  __int64 v13; // rcx
  unsigned int v14; // r8d
  int *v15; // rax
  int v16; // edi
  __int64 v17; // rdx
  _DWORD *v18; // rbx
  __int64 v19; // rdx
  __int64 *v20; // r8
  char *v21; // rdi
  int v22; // edx
  __int64 v23; // r11
  unsigned __int64 result; // rax
  int v25; // r13d
  __int16 *i; // r14
  unsigned int v27; // esi
  __int64 v28; // r9
  unsigned int v29; // edi
  unsigned __int64 v30; // rax
  int v31; // ecx
  __int64 v32; // rsi
  _DWORD *v33; // rdx
  unsigned __int64 v34; // r15
  unsigned int *v35; // rcx
  __int64 v36; // rdi
  int v37; // edx
  int v38; // eax
  int v39; // eax
  __int64 v40; // rdx
  int v41; // eax
  int v42; // eax
  unsigned __int64 v43; // rdx
  int v44; // ecx
  int v45; // ecx
  __int64 v46; // r8
  __int64 v47; // rsi
  int v48; // edi
  int v49; // r11d
  int *v50; // r10
  int v51; // ecx
  int v52; // ecx
  unsigned int v53; // esi
  int v54; // edi
  int v55; // r15d
  __int64 v56; // r10
  int v57; // ecx
  int v58; // ecx
  __int64 v59; // rdi
  unsigned int v60; // r15d
  int v61; // r10d
  int v62; // esi
  int v63; // esi
  int v64; // esi
  __int64 v65; // r8
  __int64 v66; // rcx
  int v67; // r11d
  int v68; // edi
  unsigned int v70; // [rsp+8h] [rbp-E8h]
  unsigned int v71; // [rsp+Ch] [rbp-E4h]
  int v73; // [rsp+18h] [rbp-D8h]
  __int64 v74; // [rsp+18h] [rbp-D8h]
  __int64 v75; // [rsp+18h] [rbp-D8h]
  __int64 v76; // [rsp+18h] [rbp-D8h]
  _QWORD v77[4]; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v78; // [rsp+40h] [rbp-B0h]
  __int64 v79; // [rsp+48h] [rbp-A8h]
  __int64 v80; // [rsp+50h] [rbp-A0h] BYREF
  _BYTE *v81; // [rsp+58h] [rbp-98h]
  __int64 v82; // [rsp+60h] [rbp-90h]
  int v83; // [rsp+68h] [rbp-88h]
  char v84; // [rsp+6Ch] [rbp-84h]
  _BYTE v85[32]; // [rsp+70h] [rbp-80h] BYREF
  char *v86[2]; // [rsp+90h] [rbp-60h] BYREF
  _BYTE v87[16]; // [rsp+A0h] [rbp-50h] BYREF
  char v88; // [rsp+B0h] [rbp-40h]

  sub_2E44C10((__int64)v77, a2, a4, a5);
  v71 = *(_DWORD *)(v77[1] + 8LL);
  v70 = *(_DWORD *)(v77[0] + 8LL);
  v7 = *(_DWORD *)(*(_QWORD *)(a3 + 8) + 24LL * v70 + 16) & 0xFFF;
  v8 = (__int16 *)(*(_QWORD *)(a3 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(a3 + 8) + 24LL * v70 + 16) >> 12));
  while ( v8 )
  {
    v9 = *(_DWORD *)(a1 + 24);
    v79 = 0;
    v80 = 0;
    v78 = a2;
    v86[0] = v87;
    v81 = v85;
    v82 = 4;
    v83 = 0;
    v84 = 1;
    v86[1] = (char *)0x400000000LL;
    v88 = 1;
    if ( !v9 )
    {
      ++*(_QWORD *)a1;
      goto LABEL_67;
    }
    v10 = *(_QWORD *)(a1 + 8);
    v11 = 0;
    v12 = 37 * v7;
    v13 = 1;
    v14 = (v9 - 1) & (37 * v7);
    v15 = (int *)(v10 + ((unsigned __int64)v14 << 7));
    v16 = *v15;
    if ( v7 != *v15 )
    {
      while ( v16 != -1 )
      {
        if ( v16 == -2 && !v11 )
          v11 = v15;
        v14 = (v9 - 1) & (v13 + v14);
        v15 = (int *)(v10 + ((unsigned __int64)v14 << 7));
        v16 = *v15;
        if ( v7 == *v15 )
          goto LABEL_5;
        v13 = (unsigned int)(v13 + 1);
      }
      if ( !v11 )
        v11 = v15;
      v38 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v39 = v38 + 1;
      if ( 4 * v39 < 3 * v9 )
      {
        if ( v9 - *(_DWORD *)(a1 + 20) - v39 <= v9 >> 3 )
        {
          sub_2E454E0(a1, v9);
          v63 = *(_DWORD *)(a1 + 24);
          if ( !v63 )
          {
LABEL_116:
            ++*(_DWORD *)(a1 + 16);
            BUG();
          }
          v64 = v63 - 1;
          v65 = *(_QWORD *)(a1 + 8);
          LODWORD(v66) = v64 & v12;
          v67 = 1;
          v50 = 0;
          v39 = *(_DWORD *)(a1 + 16) + 1;
          v11 = (int *)(v65 + ((unsigned __int64)(v64 & (unsigned int)v12) << 7));
          v68 = *v11;
          if ( v7 != *v11 )
          {
            while ( v68 != -1 )
            {
              if ( v68 == -2 && !v50 )
                v50 = v11;
              v10 = (unsigned int)(v67 + 1);
              v66 = v64 & (unsigned int)(v66 + v67);
              v11 = (int *)(v65 + (v66 << 7));
              v68 = *v11;
              if ( v7 == *v11 )
                goto LABEL_37;
              ++v67;
            }
            goto LABEL_71;
          }
        }
        goto LABEL_37;
      }
LABEL_67:
      sub_2E454E0(a1, 2 * v9);
      v44 = *(_DWORD *)(a1 + 24);
      if ( !v44 )
        goto LABEL_116;
      v45 = v44 - 1;
      v46 = *(_QWORD *)(a1 + 8);
      LODWORD(v47) = v45 & (37 * v7);
      v39 = *(_DWORD *)(a1 + 16) + 1;
      v11 = (int *)(v46 + ((unsigned __int64)(unsigned int)v47 << 7));
      v48 = *v11;
      if ( *v11 != v7 )
      {
        v49 = 1;
        v50 = 0;
        while ( v48 != -1 )
        {
          if ( !v50 && v48 == -2 )
            v50 = v11;
          v10 = (unsigned int)(v49 + 1);
          v47 = v45 & (unsigned int)(v47 + v49);
          v11 = (int *)(v46 + (v47 << 7));
          v48 = *v11;
          if ( v7 == *v11 )
            goto LABEL_37;
          ++v49;
        }
LABEL_71:
        if ( v50 )
          v11 = v50;
      }
LABEL_37:
      *(_DWORD *)(a1 + 16) = v39;
      if ( *v11 != -1 )
        --*(_DWORD *)(a1 + 20);
      *v11 = v7;
      memset(v11 + 2, 0, 0x78u);
      v13 = 0;
      *((_BYTE *)v11 + 52) = 1;
      v18 = v11 + 2;
      *((_QWORD *)v11 + 4) = v11 + 14;
      *((_QWORD *)v11 + 11) = v11 + 26;
      *((_QWORD *)v11 + 5) = 4;
      *((_QWORD *)v11 + 12) = 0x400000000LL;
      v17 = v78;
      goto LABEL_6;
    }
LABEL_5:
    v17 = a2;
    v18 = v15 + 2;
LABEL_6:
    *(_QWORD *)v18 = v17;
    v19 = v79;
    v20 = &v80;
    *((_QWORD *)v18 + 1) = v79;
    if ( v18 + 4 != (_DWORD *)&v80 )
      sub_C8CF80((__int64)(v18 + 4), v18 + 12, 4, (__int64)v85, (__int64)&v80);
    sub_2E44AB0((__int64)(v18 + 20), v86, v19, v13, (__int64)v20, v10);
    v21 = v86[0];
    *((_BYTE *)v18 + 112) = v88;
    if ( v21 != v87 )
      _libc_free((unsigned __int64)v21);
    if ( !v84 )
      _libc_free((unsigned __int64)v81);
    v22 = *v8++;
    v7 += v22;
    if ( !(_WORD)v22 )
      break;
  }
  v23 = a2;
  result = *(_DWORD *)(*(_QWORD *)(a3 + 8) + 24LL * v71 + 16) >> 12;
  v25 = *(_DWORD *)(*(_QWORD *)(a3 + 8) + 24LL * v71 + 16) & 0xFFF;
  for ( i = (__int16 *)(*(_QWORD *)(a3 + 56) + 2 * result); i; ++i )
  {
    v27 = *(_DWORD *)(a1 + 24);
    if ( !v27 )
    {
      ++*(_QWORD *)a1;
      goto LABEL_75;
    }
    v28 = *(_QWORD *)(a1 + 8);
    v29 = (v27 - 1) & (37 * v25);
    v30 = v28 + ((unsigned __int64)v29 << 7);
    v31 = *(_DWORD *)v30;
    if ( v25 != *(_DWORD *)v30 )
    {
      v73 = 1;
      v40 = 0;
      while ( v31 != -1 )
      {
        if ( v31 == -2 && !v40 )
          v40 = v30;
        v29 = (v27 - 1) & (v73 + v29);
        v28 = (unsigned int)(v73 + 1);
        v30 = *(_QWORD *)(a1 + 8) + ((unsigned __int64)v29 << 7);
        v31 = *(_DWORD *)v30;
        if ( v25 == *(_DWORD *)v30 )
          goto LABEL_17;
        ++v73;
      }
      if ( !v40 )
        v40 = v30;
      v41 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v42 = v41 + 1;
      if ( 4 * v42 < 3 * v27 )
      {
        if ( v27 - *(_DWORD *)(a1 + 20) - v42 <= v27 >> 3 )
        {
          v76 = v23;
          sub_2E454E0(a1, v27);
          v57 = *(_DWORD *)(a1 + 24);
          if ( !v57 )
          {
LABEL_117:
            ++*(_DWORD *)(a1 + 16);
            BUG();
          }
          v58 = v57 - 1;
          v59 = *(_QWORD *)(a1 + 8);
          v28 = 0;
          v60 = v58 & (37 * v25);
          v61 = 1;
          v23 = v76;
          v42 = *(_DWORD *)(a1 + 16) + 1;
          v40 = v59 + ((unsigned __int64)v60 << 7);
          v62 = *(_DWORD *)v40;
          if ( v25 != *(_DWORD *)v40 )
          {
            while ( v62 != -1 )
            {
              if ( v62 == -2 && !v28 )
                v28 = v40;
              v60 = v58 & (v61 + v60);
              v40 = v59 + ((unsigned __int64)v60 << 7);
              v62 = *(_DWORD *)v40;
              if ( v25 == *(_DWORD *)v40 )
                goto LABEL_46;
              ++v61;
            }
            if ( v28 )
              v40 = v28;
          }
        }
        goto LABEL_46;
      }
LABEL_75:
      v75 = v23;
      sub_2E454E0(a1, 2 * v27);
      v51 = *(_DWORD *)(a1 + 24);
      if ( !v51 )
        goto LABEL_117;
      v52 = v51 - 1;
      v28 = *(_QWORD *)(a1 + 8);
      v23 = v75;
      v53 = v52 & (37 * v25);
      v42 = *(_DWORD *)(a1 + 16) + 1;
      v40 = v28 + ((unsigned __int64)v53 << 7);
      v54 = *(_DWORD *)v40;
      if ( v25 != *(_DWORD *)v40 )
      {
        v55 = 1;
        v56 = 0;
        while ( v54 != -1 )
        {
          if ( !v56 && v54 == -2 )
            v56 = v40;
          v53 = v52 & (v55 + v53);
          v40 = v28 + ((unsigned __int64)v53 << 7);
          v54 = *(_DWORD *)v40;
          if ( v25 == *(_DWORD *)v40 )
            goto LABEL_46;
          ++v55;
        }
        if ( v56 )
          v40 = v56;
      }
LABEL_46:
      *(_DWORD *)(a1 + 16) = v42;
      if ( *(_DWORD *)v40 != -1 )
        --*(_DWORD *)(a1 + 20);
      *(_DWORD *)v40 = v25;
      memset((void *)(v40 + 8), 0, 0x78u);
      v35 = (unsigned int *)(v40 + 104);
      *(_BYTE *)(v40 + 52) = 1;
      *(_QWORD *)(v40 + 32) = v40 + 56;
      v34 = v40 + 8;
      v32 = 0;
      *(_QWORD *)(v40 + 40) = 4;
      *(_QWORD *)(v40 + 88) = v40 + 104;
      *(_QWORD *)(v40 + 96) = 0x400000000LL;
LABEL_49:
      result = *(unsigned int *)(v34 + 92);
      v43 = v32 + 1;
      if ( v32 + 1 > result )
      {
LABEL_60:
        v74 = v23;
        sub_C8D5F0(v34 + 80, (const void *)(v34 + 96), v43, 4u, 0x400000000LL, v28);
        result = *(_QWORD *)(v34 + 80);
        v23 = v74;
        v35 = (unsigned int *)(result + 4LL * *(unsigned int *)(v34 + 88));
      }
LABEL_50:
      *v35 = v70;
      ++*(_DWORD *)(v34 + 88);
      goto LABEL_25;
    }
LABEL_17:
    v32 = *(unsigned int *)(v30 + 96);
    v33 = *(_DWORD **)(v30 + 88);
    v34 = v30 + 8;
    v35 = &v33[v32];
    v36 = (4 * v32) >> 2;
    result = (4 * v32) >> 4;
    if ( result )
    {
      while ( v70 != *v33 )
      {
        if ( v70 == v33[1] )
        {
          ++v33;
          goto LABEL_24;
        }
        if ( v70 == v33[2] )
        {
          v33 += 2;
          goto LABEL_24;
        }
        if ( v70 == v33[3] )
        {
          v33 += 3;
          goto LABEL_24;
        }
        v33 += 4;
        if ( !--result )
        {
          v36 = v35 - v33;
          goto LABEL_55;
        }
      }
      goto LABEL_24;
    }
LABEL_55:
    if ( v36 == 2 )
      goto LABEL_56;
    if ( v36 == 3 )
    {
      if ( v70 == *v33 )
        goto LABEL_24;
      ++v33;
LABEL_56:
      if ( v70 == *v33 )
        goto LABEL_24;
      ++v33;
      goto LABEL_58;
    }
    if ( v36 != 1 )
      goto LABEL_49;
LABEL_58:
    if ( v70 != *v33 )
    {
      result = *(unsigned int *)(v34 + 92);
      v43 = v32 + 1;
      if ( v32 + 1 > result )
        goto LABEL_60;
      goto LABEL_50;
    }
LABEL_24:
    if ( v35 == v33 )
      goto LABEL_49;
LABEL_25:
    *(_QWORD *)(v34 + 8) = v23;
    v37 = *i;
    v25 += v37;
    if ( !(_WORD)v37 )
      return result;
  }
  return result;
}
