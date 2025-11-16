// Function: sub_2FFC100
// Address: 0x2ffc100
//
__int64 __fastcall sub_2FFC100(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  unsigned int v4; // r10d
  __int64 v7; // rdi
  int v8; // esi
  unsigned int v9; // ecx
  _DWORD *v10; // r10
  int v11; // r11d
  __int64 v12; // rdx
  _DWORD *v13; // r15
  unsigned __int64 v14; // r12
  __int64 v15; // r15
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // r12
  __int64 v21; // r11
  _DWORD *v22; // rax
  int v23; // r15d
  int v24; // esi
  unsigned __int8 v25; // r10
  char v26; // dl
  unsigned int v27; // esi
  __int64 v29; // rcx
  _DWORD *v30; // rax
  unsigned int v31; // edi
  unsigned int v32; // r10d
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rsi
  int v36; // edx
  __int64 v37; // rcx
  int v38; // edi
  __int64 v39; // rsi
  int v40; // edx
  __int64 v41; // rcx
  int v42; // edi
  int v43; // r10d
  _DWORD *v44; // r9
  int v45; // edx
  int v46; // edx
  int v47; // r10d
  __int64 v49; // [rsp+10h] [rbp-50h]
  unsigned __int8 v50; // [rsp+10h] [rbp-50h]
  unsigned __int8 v51; // [rsp+18h] [rbp-48h]
  __int64 v52; // [rsp+18h] [rbp-48h]
  __int64 v53; // [rsp+18h] [rbp-48h]
  unsigned int v54; // [rsp+24h] [rbp-3Ch]
  __int64 v55; // [rsp+28h] [rbp-38h]
  _DWORD *v56; // [rsp+28h] [rbp-38h]
  _DWORD *v57; // [rsp+28h] [rbp-38h]
  __int64 v58; // [rsp+28h] [rbp-38h]
  __int64 v59; // [rsp+28h] [rbp-38h]
  unsigned int v60; // [rsp+28h] [rbp-38h]

  v54 = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
  if ( v54 )
  {
    v3 = 0;
    v4 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v15 = 40 * v3;
        v16 = 40 * v3 + *(_QWORD *)(a2 + 32);
        if ( *(_BYTE *)v16 || (*(_BYTE *)(v16 + 3) & 0x10) != 0 || (*(_WORD *)(v16 + 2) & 0xFF0) == 0 )
          goto LABEL_8;
        v55 = a3;
        v17 = sub_2E89F40(a2, v3);
        v19 = *(_QWORD *)(a2 + 32);
        a3 = v55;
        v20 = v17;
        v21 = v19 + v15;
        v22 = (_DWORD *)(v19 + 40LL * v17);
        v23 = *(_DWORD *)(v19 + v15 + 8);
        v24 = v22[2];
        if ( v23 != v24 )
          break;
        v4 = 1;
        if ( v54 <= (unsigned int)++v3 )
          return v4;
      }
      v25 = *(_BYTE *)(v21 + 4) & 1;
      if ( v25 && (*v22 & 0xFFF00) == 0 )
      {
        if ( v24 < 0 )
        {
          v50 = *(_BYTE *)(v21 + 4) & 1;
          v53 = v21;
          sub_2EBE590(
            *(_QWORD *)(a1 + 32),
            v24,
            *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 56LL) + 16LL * (v23 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
            0);
          a3 = v55;
          v25 = v50;
          v21 = v53;
        }
        v49 = a3;
        v51 = v25;
        v56 = (_DWORD *)v21;
        sub_2EAB0C0(v21, v24);
        *v56 &= 0xFFF000FF;
        v4 = v51;
        a3 = v49;
        goto LABEL_8;
      }
      v26 = *(_BYTE *)(v55 + 8) & 1;
      if ( v26 )
      {
        v7 = v55 + 16;
        v8 = 3;
      }
      else
      {
        v27 = *(_DWORD *)(v55 + 24);
        v7 = *(_QWORD *)(v55 + 16);
        if ( !v27 )
        {
          v29 = *(unsigned int *)(v55 + 8);
          ++*(_QWORD *)v55;
          v30 = 0;
          v31 = ((unsigned int)v29 >> 1) + 1;
          goto LABEL_24;
        }
        v8 = v27 - 1;
      }
      v9 = v8 & (37 * v23);
      v10 = (_DWORD *)(v7 + 56LL * v9);
      v11 = *v10;
      if ( v23 != *v10 )
        break;
LABEL_5:
      v12 = (unsigned int)v10[4];
      v13 = v10 + 2;
      v14 = (unsigned int)v3 | (unsigned __int64)(v20 << 32);
      if ( v12 + 1 > (unsigned __int64)(unsigned int)v10[5] )
      {
        v52 = a3;
        v57 = v10;
        sub_C8D5F0((__int64)(v10 + 2), v10 + 6, v12 + 1, 8u, a3, v18);
        a3 = v52;
        v12 = (unsigned int)v57[4];
      }
LABEL_7:
      v4 = 1;
      *(_QWORD *)(*(_QWORD *)v13 + 8 * v12) = v14;
      ++v13[2];
LABEL_8:
      if ( v54 <= (unsigned int)++v3 )
        return v4;
    }
    v18 = 1;
    v30 = 0;
    while ( v11 != -1 )
    {
      if ( v11 == -2 && !v30 )
        v30 = v10;
      v9 = v8 & (v18 + v9);
      v60 = v18 + 1;
      v18 = 7LL * v9;
      v10 = (_DWORD *)(v7 + 56LL * v9);
      v11 = *v10;
      if ( v23 == *v10 )
        goto LABEL_5;
      v18 = v60;
    }
    v29 = *(unsigned int *)(a3 + 8);
    v27 = 4;
    if ( !v30 )
      v30 = v10;
    ++*(_QWORD *)a3;
    v32 = 12;
    v31 = ((unsigned int)v29 >> 1) + 1;
    if ( !v26 )
    {
      v27 = *(_DWORD *)(a3 + 24);
LABEL_24:
      v32 = 3 * v27;
    }
    v33 = 4 * v31;
    if ( (unsigned int)v33 >= v32 )
    {
      v58 = a3;
      sub_2FFBB70(a3, 2 * v27, v33, v29, a3, v18);
      a3 = v58;
      if ( (*(_BYTE *)(v58 + 8) & 1) != 0 )
      {
        v35 = v58 + 16;
        v36 = 3;
      }
      else
      {
        v45 = *(_DWORD *)(v58 + 24);
        v35 = *(_QWORD *)(v58 + 16);
        if ( !v45 )
          goto LABEL_69;
        v36 = v45 - 1;
      }
      LODWORD(v37) = v36 & (37 * v23);
      v30 = (_DWORD *)(v35 + 56LL * (unsigned int)v37);
      v38 = *v30;
      if ( v23 == *v30 )
        goto LABEL_39;
      v47 = 1;
      v44 = 0;
      while ( v38 != -1 )
      {
        if ( !v44 && v38 == -2 )
          v44 = v30;
        v37 = v36 & (unsigned int)(v37 + v47);
        v30 = (_DWORD *)(v35 + 56 * v37);
        v38 = *v30;
        if ( v23 == *v30 )
          goto LABEL_39;
        ++v47;
      }
    }
    else
    {
      v34 = v27 - *(_DWORD *)(a3 + 12) - v31;
      if ( (unsigned int)v34 > v27 >> 3 )
      {
LABEL_27:
        *(_DWORD *)(a3 + 8) = (2 * ((unsigned int)v29 >> 1) + 2) | v29 & 1;
        if ( *v30 != -1 )
          --*(_DWORD *)(a3 + 12);
        *v30 = v23;
        v13 = v30 + 2;
        *((_QWORD *)v30 + 1) = v30 + 6;
        v14 = (unsigned int)v3 | (unsigned __int64)(v20 << 32);
        v12 = 0;
        *((_QWORD *)v30 + 2) = 0x400000000LL;
        goto LABEL_7;
      }
      v59 = a3;
      sub_2FFBB70(a3, v27, v34, v29, a3, v18);
      a3 = v59;
      if ( (*(_BYTE *)(v59 + 8) & 1) != 0 )
      {
        v39 = v59 + 16;
        v40 = 3;
      }
      else
      {
        v46 = *(_DWORD *)(v59 + 24);
        v39 = *(_QWORD *)(v59 + 16);
        if ( !v46 )
        {
LABEL_69:
          *(_DWORD *)(a3 + 8) = (2 * (*(_DWORD *)(a3 + 8) >> 1) + 2) | *(_DWORD *)(a3 + 8) & 1;
          BUG();
        }
        v40 = v46 - 1;
      }
      LODWORD(v41) = v40 & (37 * v23);
      v30 = (_DWORD *)(v39 + 56LL * (unsigned int)v41);
      v42 = *v30;
      if ( v23 == *v30 )
      {
LABEL_39:
        LODWORD(v29) = *(_DWORD *)(a3 + 8);
        goto LABEL_27;
      }
      v43 = 1;
      v44 = 0;
      while ( v42 != -1 )
      {
        if ( !v44 && v42 == -2 )
          v44 = v30;
        v41 = v40 & (unsigned int)(v41 + v43);
        v30 = (_DWORD *)(v39 + 56 * v41);
        v42 = *v30;
        if ( v23 == *v30 )
          goto LABEL_39;
        ++v43;
      }
    }
    if ( v44 )
      v30 = v44;
    goto LABEL_39;
  }
  return 0;
}
