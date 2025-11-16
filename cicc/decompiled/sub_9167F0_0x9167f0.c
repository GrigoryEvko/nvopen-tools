// Function: sub_9167F0
// Address: 0x9167f0
//
__int64 __fastcall sub_9167F0(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r9
  unsigned int v9; // esi
  __int64 v10; // rdi
  int v11; // r11d
  __int64 *v12; // rbx
  unsigned int v13; // r15d
  unsigned int v14; // ecx
  _QWORD *v15; // rdx
  __int64 v16; // rax
  const __m128i *v17; // r15
  const __m128i **v18; // rbx
  __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // r8
  __int64 v22; // r9
  const __m128i *v23; // rdi
  int v25; // eax
  int v26; // edx
  __int64 v27; // r15
  size_t v28; // rax
  __int64 v29; // rax
  char v30; // al
  unsigned int v31; // r11d
  unsigned int v32; // eax
  char v33; // al
  __int64 v34; // rsi
  __int64 v35; // rax
  const char *v36; // r10
  size_t v37; // rax
  __int64 v38; // rcx
  const char *v39; // r10
  size_t v40; // r8
  _QWORD *v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // rsi
  unsigned __int16 v44; // di
  _BOOL4 v45; // eax
  __int64 v46; // rax
  _QWORD *v47; // rdi
  __int64 v48; // rax
  int v49; // eax
  int v50; // ecx
  __int64 v51; // rsi
  unsigned int v52; // eax
  __int64 v53; // rdi
  int v54; // r9d
  __int64 *v55; // r8
  int v56; // eax
  int v57; // eax
  __int64 v58; // rsi
  int v59; // r8d
  unsigned int v60; // r15d
  __int64 *v61; // rdi
  __int64 v62; // rcx
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rax
  char *v66; // rax
  size_t n; // [rsp+0h] [rbp-D0h]
  unsigned int src; // [rsp+8h] [rbp-C8h]
  const char *v69; // [rsp+10h] [rbp-C0h]
  __int64 v70; // [rsp+18h] [rbp-B8h]
  char *v71; // [rsp+18h] [rbp-B8h]
  char *s; // [rsp+20h] [rbp-B0h]
  char *sa; // [rsp+20h] [rbp-B0h]
  unsigned int sb; // [rsp+20h] [rbp-B0h]
  _QWORD *v75; // [rsp+28h] [rbp-A8h]
  char *v76; // [rsp+28h] [rbp-A8h]
  char *v77; // [rsp+28h] [rbp-A8h]
  char *v78; // [rsp+28h] [rbp-A8h]
  __int64 v79; // [rsp+28h] [rbp-A8h]
  __int64 v80; // [rsp+28h] [rbp-A8h]
  unsigned int v81; // [rsp+28h] [rbp-A8h]
  char *v82; // [rsp+28h] [rbp-A8h]
  char *v83; // [rsp+28h] [rbp-A8h]
  _QWORD v84[2]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD v85[2]; // [rsp+40h] [rbp-90h] BYREF
  _QWORD v86[2]; // [rsp+50h] [rbp-80h] BYREF
  __int64 v87; // [rsp+60h] [rbp-70h] BYREF
  _QWORD v88[4]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v89; // [rsp+90h] [rbp-40h]

  v5 = a1 + 376;
  v9 = *(_DWORD *)(a1 + 400);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 376);
    goto LABEL_57;
  }
  v10 = *(_QWORD *)(a1 + 384);
  v11 = 1;
  v12 = 0;
  v13 = ((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4);
  v14 = (v9 - 1) & v13;
  v15 = (_QWORD *)(v10 + 16LL * v14);
  v16 = *v15;
  if ( a4 != *v15 )
  {
    while ( v16 != -4096 )
    {
      if ( v16 == -8192 && !v12 )
        v12 = v15;
      v14 = (v9 - 1) & (v11 + v14);
      v15 = (_QWORD *)(v10 + 16LL * v14);
      v16 = *v15;
      if ( a4 == *v15 )
        goto LABEL_3;
      ++v11;
    }
    v25 = *(_DWORD *)(a1 + 392);
    if ( !v12 )
      v12 = v15;
    ++*(_QWORD *)(a1 + 376);
    v26 = v25 + 1;
    if ( 4 * (v25 + 1) < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(a1 + 396) - v26 > v9 >> 3 )
      {
LABEL_17:
        *(_DWORD *)(a1 + 392) = v26;
        if ( *v12 != -4096 )
          --*(_DWORD *)(a1 + 396);
        *v12 = a4;
        v18 = (const __m128i **)(v12 + 1);
        *v18 = 0;
        goto LABEL_20;
      }
      v83 = a2;
      sub_915A60(v5, v9);
      v56 = *(_DWORD *)(a1 + 400);
      if ( v56 )
      {
        v57 = v56 - 1;
        v58 = *(_QWORD *)(a1 + 384);
        v59 = 1;
        v60 = v57 & v13;
        a2 = v83;
        v26 = *(_DWORD *)(a1 + 392) + 1;
        v61 = 0;
        v12 = (__int64 *)(v58 + 16LL * v60);
        v62 = *v12;
        if ( a4 != *v12 )
        {
          while ( v62 != -4096 )
          {
            if ( !v61 && v62 == -8192 )
              v61 = v12;
            v60 = v57 & (v59 + v60);
            v12 = (__int64 *)(v58 + 16LL * v60);
            v62 = *v12;
            if ( a4 == *v12 )
              goto LABEL_17;
            ++v59;
          }
          if ( v61 )
            v12 = v61;
        }
        goto LABEL_17;
      }
LABEL_94:
      ++*(_DWORD *)(a1 + 392);
      BUG();
    }
LABEL_57:
    v82 = a2;
    sub_915A60(v5, 2 * v9);
    v49 = *(_DWORD *)(a1 + 400);
    if ( v49 )
    {
      v50 = v49 - 1;
      v51 = *(_QWORD *)(a1 + 384);
      a2 = v82;
      v52 = (v49 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
      v26 = *(_DWORD *)(a1 + 392) + 1;
      v12 = (__int64 *)(v51 + 16LL * v52);
      v53 = *v12;
      if ( a4 != *v12 )
      {
        v54 = 1;
        v55 = 0;
        while ( v53 != -4096 )
        {
          if ( !v55 && v53 == -8192 )
            v55 = v12;
          v52 = v50 & (v54 + v52);
          v12 = (__int64 *)(v51 + 16LL * v52);
          v53 = *v12;
          if ( a4 == *v12 )
            goto LABEL_17;
          ++v54;
        }
        if ( v55 )
          v12 = v55;
      }
      goto LABEL_17;
    }
    goto LABEL_94;
  }
LABEL_3:
  v17 = (const __m128i *)v15[1];
  v18 = (const __m128i **)(v15 + 1);
  if ( v17 )
  {
    v75 = v15;
    if ( a3 != v17[1].m128i_i64[1] )
    {
      v20 = sub_BCE760(a3, 0);
      v23 = (const __m128i *)v75[1];
      return sub_AD4C90(v23, v20, 0, v19, v21, v22);
    }
    return (__int64)v17;
  }
LABEL_20:
  if ( a2 )
  {
    if ( !memcmp(a2, "__builtin_", 0xAu) )
    {
      v77 = a2;
      v30 = sub_915490(a2);
      a2 = v77;
      if ( !v30 )
        a2 = v77 + 10;
    }
    v27 = *(_QWORD *)a1;
    v76 = a2;
    v28 = strlen(a2);
    v29 = sub_BA8CB0(v27, v76, v28);
    a2 = v76;
    v17 = (const __m128i *)v29;
    if ( v29 )
    {
LABEL_23:
      *v18 = v17;
      if ( a3 != v17[1].m128i_i64[1] )
      {
        v23 = v17;
        v20 = sub_BCE760(a3, 0);
        return sub_AD4C90(v23, v20, 0, v19, v21, v22);
      }
      return (__int64)v17;
    }
  }
  v31 = 0;
  if ( (*(_BYTE *)(a4 - 8) & 0x10) == 0 )
  {
    v78 = a2;
    v32 = sub_909290(a4, unk_4D046B4 != 0);
    a2 = v78;
    v31 = v32;
  }
  if ( dword_4D04530 && (*(_BYTE *)(a4 + 193) & 2) != 0
    || !*(_BYTE *)(a4 + 174)
    && (v44 = *(_WORD *)(a4 + 176)) != 0
    && (sa = a2, v81 = v31, v45 = sub_825D70(v44), v31 = v81, a2 = sa, v45) )
  {
    v33 = *(_BYTE *)(a4 + 198);
    goto LABEL_35;
  }
  v33 = *(_BYTE *)(a4 + 198);
  if ( (v33 & 0x10) != 0 )
  {
LABEL_35:
    v34 = a3;
    v79 = a1 + 8;
    if ( (v33 & 0x20) != 0 && (*(_BYTE *)(a4 + 199) & 8) != 0 )
    {
      v71 = a2;
      sb = v31;
      v63 = sub_91A390(v79, *(_QWORD *)(a4 + 152), 1);
      a2 = v71;
      v31 = sb;
      v34 = v63;
    }
    s = a2;
    src = v31;
    v70 = *(_QWORD *)a1;
    v89 = 257;
    v35 = sub_BD2DA0(136);
    v36 = s;
    v17 = (const __m128i *)v35;
    if ( v35 )
    {
      sub_B2C3B0(v35, v34, src, 0xFFFFFFFFLL, v88, v70);
      v36 = s;
    }
    if ( !v36 )
      goto LABEL_46;
    v69 = v36;
    v84[0] = v85;
    v37 = strlen(v36);
    v39 = v69;
    v88[0] = v37;
    v40 = v37;
    if ( v37 > 0xF )
    {
      n = v37;
      v46 = sub_22409D0(v84, v88, 0);
      v39 = v69;
      v40 = n;
      v84[0] = v46;
      v47 = (_QWORD *)v46;
      v85[0] = v88[0];
    }
    else
    {
      if ( v37 == 1 )
      {
        LOBYTE(v85[0]) = *v69;
        v41 = v85;
LABEL_42:
        v84[1] = v37;
        *((_BYTE *)v41 + v37) = 0;
        sub_91C110(v86, v84, a4, v38, v40);
        v89 = 260;
        v88[0] = v86;
        sub_BD6B50(v17, v88);
        if ( (__int64 *)v86[0] != &v87 )
          j_j___libc_free_0(v86[0], v87 + 1);
        if ( (_QWORD *)v84[0] != v85 )
          j_j___libc_free_0(v84[0], v85[0] + 1LL);
LABEL_46:
        v42 = 0;
        if ( (*(_BYTE *)(a4 + 198) & 0x20) != 0 && (v48 = *(_QWORD *)(a4 + 128), v42 = 1, v48) )
        {
          v43 = *(_QWORD *)(v48 + 152);
          v42 = 1;
        }
        else
        {
          v43 = *(_QWORD *)(a4 + 152);
        }
        v80 = sub_9380F0(v79, v43, v42);
        sub_90A4E0(a1, a4, v80, (__int64)v17);
        sub_913A20((_QWORD *)a1, a4, v80, v17);
        goto LABEL_23;
      }
      if ( !v37 )
      {
        v41 = v85;
        goto LABEL_42;
      }
      v47 = v85;
    }
    memcpy(v47, v39, v40);
    v37 = v88[0];
    v41 = (_QWORD *)v84[0];
    goto LABEL_42;
  }
  if ( qword_4F04C50 )
  {
    v64 = *(_QWORD *)(qword_4F04C50 + 32LL);
    if ( v64 && ((*(_BYTE *)(v64 + 198) & 0x18) == 0x18 || (*(_BYTE *)(v64 + 197) & 4) != 0)
      || (*(_BYTE *)(a4 + 193) & 0x10) != 0
      || (*(_BYTE *)(v64 + 198) & 0x18) != 0x10 )
    {
      goto LABEL_78;
    }
    goto LABEL_82;
  }
  if ( (*(_BYTE *)(a4 + 193) & 0x10) == 0 )
  {
LABEL_82:
    v66 = sub_693CD0(a2);
    sub_6851A0(0xDC5u, dword_4F07508, (__int64)v66);
  }
LABEL_78:
  v65 = sub_BCE760(a3, 0);
  return sub_AD6530(v65);
}
