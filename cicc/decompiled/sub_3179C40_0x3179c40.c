// Function: sub_3179C40
// Address: 0x3179c40
//
__int64 __fastcall sub_3179C40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r13d
  _QWORD *v6; // rdi
  unsigned int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // rbx
  unsigned int v10; // esi
  __int64 v11; // r9
  int v12; // r15d
  _QWORD *v13; // r11
  unsigned int v14; // r8d
  _QWORD *v15; // rcx
  __int64 v16; // rdx
  unsigned int v17; // r12d
  int v18; // eax
  int v19; // edx
  __int64 v20; // r14
  __int64 v21; // rax
  _BYTE *v22; // r15
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // rsi
  unsigned int v26; // ecx
  __int64 *v27; // rdi
  __int64 v28; // r10
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  int v35; // eax
  int v36; // ecx
  __int64 v37; // rdi
  unsigned int v38; // eax
  __int64 v39; // rsi
  int v40; // r9d
  _QWORD *v41; // r8
  int v42; // eax
  int v43; // eax
  __int64 v44; // rsi
  int v45; // r8d
  unsigned int v46; // r14d
  _QWORD *v47; // rdi
  __int64 v48; // rcx
  int v49; // edi
  int v50; // edx
  unsigned int v51; // [rsp+14h] [rbp-26Ch]
  __int64 v53; // [rsp+28h] [rbp-258h]
  _QWORD *v55; // [rsp+40h] [rbp-240h] BYREF
  __int64 v56; // [rsp+48h] [rbp-238h]
  _QWORD v57[70]; // [rsp+50h] [rbp-230h] BYREF

  v4 = 0;
  v6 = v57;
  v55 = v57;
  v57[0] = a3;
  v56 = 0x4000000001LL;
  v7 = 1;
  while ( 1 )
  {
    v8 = v7;
    ++v4;
    --v7;
    v9 = v6[v8 - 1];
    LODWORD(v56) = v7;
    if ( v4 > (unsigned int)qword_5034C88 || (unsigned int)qword_5034BA8 < (*(_DWORD *)(v9 + 4) & 0x7FFFFFFu) )
      break;
    v10 = *(_DWORD *)(a4 + 24);
    if ( !v10 )
    {
      ++*(_QWORD *)a4;
      goto LABEL_43;
    }
    v11 = *(_QWORD *)(a4 + 8);
    v12 = 1;
    v13 = 0;
    v14 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v15 = (_QWORD *)(v11 + 8LL * v14);
    v16 = *v15;
    if ( v9 != *v15 )
    {
      while ( v16 != -4096 )
      {
        if ( v16 != -8192 || v13 )
          v15 = v13;
        v14 = (v10 - 1) & (v12 + v14);
        v16 = *(_QWORD *)(v11 + 8LL * v14);
        if ( v9 == v16 )
          goto LABEL_6;
        ++v12;
        v13 = v15;
        v15 = (_QWORD *)(v11 + 8LL * v14);
      }
      v18 = *(_DWORD *)(a4 + 16);
      if ( !v13 )
        v13 = v15;
      ++*(_QWORD *)a4;
      v19 = v18 + 1;
      if ( 4 * (v18 + 1) < 3 * v10 )
      {
        if ( v10 - *(_DWORD *)(a4 + 20) - v19 > v10 >> 3 )
          goto LABEL_17;
        sub_110B120(a4, v10);
        v42 = *(_DWORD *)(a4 + 24);
        if ( v42 )
        {
          v43 = v42 - 1;
          v44 = *(_QWORD *)(a4 + 8);
          v45 = 1;
          v46 = v43 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v13 = (_QWORD *)(v44 + 8LL * v46);
          v19 = *(_DWORD *)(a4 + 16) + 1;
          v47 = 0;
          v48 = *v13;
          if ( v9 != *v13 )
          {
            while ( v48 != -4096 )
            {
              if ( !v47 && v48 == -8192 )
                v47 = v13;
              v46 = v43 & (v45 + v46);
              v13 = (_QWORD *)(v44 + 8LL * v46);
              v48 = *v13;
              if ( v9 == *v13 )
                goto LABEL_17;
              ++v45;
            }
            if ( v47 )
              v13 = v47;
          }
          goto LABEL_17;
        }
LABEL_72:
        ++*(_DWORD *)(a4 + 16);
        BUG();
      }
LABEL_43:
      sub_110B120(a4, 2 * v10);
      v35 = *(_DWORD *)(a4 + 24);
      if ( !v35 )
        goto LABEL_72;
      v36 = v35 - 1;
      v37 = *(_QWORD *)(a4 + 8);
      v38 = (v35 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v13 = (_QWORD *)(v37 + 8LL * v38);
      v39 = *v13;
      v19 = *(_DWORD *)(a4 + 16) + 1;
      if ( v9 != *v13 )
      {
        v40 = 1;
        v41 = 0;
        while ( v39 != -4096 )
        {
          if ( v39 == -8192 && !v41 )
            v41 = v13;
          v38 = v36 & (v40 + v38);
          v13 = (_QWORD *)(v37 + 8LL * v38);
          v39 = *v13;
          if ( v9 == *v13 )
            goto LABEL_17;
          ++v40;
        }
        if ( v41 )
          v13 = v41;
      }
LABEL_17:
      *(_DWORD *)(a4 + 16) = v19;
      if ( *v13 != -4096 )
        --*(_DWORD *)(a4 + 20);
      *v13 = v9;
      if ( (*(_DWORD *)(v9 + 4) & 0x7FFFFFF) == 0 )
      {
        v7 = v56;
        v6 = v55;
        goto LABEL_6;
      }
      v51 = v4;
      v53 = 8LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF);
      v20 = 0;
      while ( 2 )
      {
        v21 = *(_QWORD *)(v9 - 8);
        v22 = *(_BYTE **)(v21 + 4 * v20);
        if ( !v22 )
          BUG();
        if ( *v22 > 0x1Cu )
        {
          if ( (_BYTE *)v9 == v22 )
            goto LABEL_25;
          v23 = *(_QWORD *)(32LL * *(unsigned int *)(v9 + 72) + v21 + v20);
          if ( !(unsigned __int8)sub_2A64220(*(__int64 **)(a1 + 56), v23) )
            goto LABEL_25;
          v24 = *(unsigned int *)(a1 + 120);
          v25 = *(_QWORD *)(a1 + 104);
          if ( (_DWORD)v24 )
          {
            v26 = (v24 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
            v27 = (__int64 *)(v25 + 8LL * v26);
            v28 = *v27;
            if ( v23 == *v27 )
            {
LABEL_29:
              if ( v27 != (__int64 *)(v25 + 8 * v24) )
                goto LABEL_25;
            }
            else
            {
              v49 = 1;
              while ( v28 != -4096 )
              {
                v50 = v49 + 1;
                v26 = (v24 - 1) & (v49 + v26);
                v27 = (__int64 *)(v25 + 8LL * v26);
                v28 = *v27;
                if ( v23 == *v27 )
                  goto LABEL_29;
                v49 = v50;
              }
            }
          }
        }
        v29 = sub_31751A0(a1, v22);
        if ( v29 )
        {
          if ( v29 != a2 )
            goto LABEL_32;
        }
        else
        {
          if ( *v22 != 84 )
          {
LABEL_32:
            v6 = v55;
            v17 = 0;
            goto LABEL_33;
          }
          v33 = (unsigned int)v56;
          v34 = (unsigned int)v56 + 1LL;
          if ( v34 > HIDWORD(v56) )
          {
            sub_C8D5F0((__int64)&v55, v57, v34, 8u, v30, v31);
            v33 = (unsigned int)v56;
          }
          v55[v33] = v22;
          LODWORD(v56) = v56 + 1;
        }
LABEL_25:
        v20 += 8;
        if ( v20 == v53 )
        {
          v4 = v51;
          v7 = v56;
          v6 = v55;
          break;
        }
        continue;
      }
    }
LABEL_6:
    if ( !v7 )
    {
      v17 = 1;
      goto LABEL_33;
    }
  }
  v17 = 0;
LABEL_33:
  if ( v6 != v57 )
    _libc_free((unsigned __int64)v6);
  return v17;
}
