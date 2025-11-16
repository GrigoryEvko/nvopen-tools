// Function: sub_1ABF1D0
// Address: 0x1abf1d0
//
__int64 __fastcall sub_1ABF1D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v7; // r15
  __int64 v8; // r12
  char **v9; // rbx
  char **v10; // r12
  __int64 v11; // r11
  int v12; // ecx
  __int64 v13; // rdi
  unsigned int v14; // edx
  char *v15; // rsi
  char *v16; // rax
  int v17; // edx
  char v18; // dl
  int v19; // edx
  __int64 v20; // rsi
  int v21; // ecx
  __int64 v22; // rdi
  unsigned int v23; // edx
  __int64 v24; // r8
  int v25; // r9d
  unsigned int v26; // esi
  __int64 v27; // r8
  unsigned int v28; // edx
  char **v29; // rdi
  char *v30; // rcx
  char **v31; // r10
  int v32; // edi
  int v33; // edx
  char **v34; // rsi
  __int64 v35; // rbx
  int v36; // edx
  __int64 v37; // rcx
  int v38; // edx
  __int64 v39; // rdi
  int v40; // r8d
  unsigned int v41; // eax
  __int64 v42; // rsi
  _QWORD *v43; // rax
  int v44; // r8d
  int v45; // edx
  int v46; // r9d
  __int64 v47; // r8
  unsigned int v48; // ecx
  char *v49; // rsi
  char **v50; // rdi
  int v51; // eax
  __int64 v52; // r9
  int v53; // r8d
  unsigned int v54; // ecx
  char **v55; // rsi
  int v56; // [rsp+10h] [rbp-80h]
  __int64 v57; // [rsp+10h] [rbp-80h]
  __int64 v58; // [rsp+10h] [rbp-80h]
  int v59; // [rsp+10h] [rbp-80h]
  __int64 v60; // [rsp+10h] [rbp-80h]
  int v61; // [rsp+10h] [rbp-80h]
  __int64 v62; // [rsp+18h] [rbp-78h]
  __int64 v64; // [rsp+28h] [rbp-68h]
  __int64 v66; // [rsp+40h] [rbp-50h]
  char *v67[7]; // [rsp+58h] [rbp-38h] BYREF

  result = *(_QWORD *)(a1 + 80);
  v62 = result;
  v64 = *(_QWORD *)(a1 + 72);
  if ( v64 != result )
  {
    while ( 1 )
    {
      v66 = *(_QWORD *)v64 + 40LL;
      v7 = *(_QWORD *)(*(_QWORD *)v64 + 48LL);
      if ( v7 != v66 )
        break;
LABEL_39:
      v64 += 8;
      result = v64;
      if ( v62 == v64 )
        return result;
    }
    while ( 1 )
    {
      if ( !v7 )
        BUG();
      v8 = 3LL * (*(_DWORD *)(v7 - 4) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v7 - 1) & 0x40) != 0 )
      {
        v9 = *(char ***)(v7 - 32);
        v10 = &v9[v8];
      }
      else
      {
        v9 = (char **)(v7 - 24 - v8 * 8);
        v10 = (char **)(v7 - 24);
      }
      v11 = a2;
      if ( v9 != v10 )
        break;
LABEL_31:
      v35 = *(_QWORD *)(v7 - 16);
      if ( v35 )
      {
        while ( 1 )
        {
          v43 = sub_1648700(v35);
          if ( *((_BYTE *)v43 + 16) <= 0x17u )
            break;
          v36 = *(_DWORD *)(a1 + 64);
          if ( !v36 )
            break;
          v37 = v43[5];
          v38 = v36 - 1;
          v39 = *(_QWORD *)(a1 + 48);
          v40 = 1;
          v41 = v38 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
          v42 = *(_QWORD *)(v39 + 8LL * v41);
          if ( v42 != v37 )
          {
            while ( v42 != -8 )
            {
              v41 = v38 & (v40 + v41);
              v42 = *(_QWORD *)(v39 + 8LL * v41);
              if ( v37 == v42 )
                goto LABEL_35;
              ++v40;
            }
            break;
          }
LABEL_35:
          v35 = *(_QWORD *)(v35 + 8);
          if ( !v35 )
            goto LABEL_38;
        }
        v67[0] = (char *)(v7 - 24);
        sub_1ABE500(a3, v67);
      }
LABEL_38:
      v7 = *(_QWORD *)(v7 + 8);
      if ( v66 == v7 )
        goto LABEL_39;
    }
    while ( 1 )
    {
      v16 = *v9;
      v17 = *(_DWORD *)(a4 + 24);
      v67[0] = *v9;
      if ( v17 )
      {
        v12 = v17 - 1;
        v13 = *(_QWORD *)(a4 + 8);
        v14 = (v17 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v15 = *(char **)(v13 + 8LL * v14);
        if ( v16 == v15 )
          goto LABEL_9;
        v44 = 1;
        while ( v15 != (char *)-8LL )
        {
          v14 = v12 & (v44 + v14);
          v15 = *(char **)(v13 + 8LL * v14);
          if ( v16 == v15 )
            goto LABEL_9;
          ++v44;
        }
      }
      v18 = v16[16];
      if ( v18 != 17 )
      {
        if ( (unsigned __int8)v18 <= 0x17u )
          goto LABEL_9;
        v19 = *(_DWORD *)(a1 + 64);
        if ( v19 )
        {
          v20 = *((_QWORD *)v16 + 5);
          v21 = v19 - 1;
          v22 = *(_QWORD *)(a1 + 48);
          v23 = (v19 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          v24 = *(_QWORD *)(v22 + 8LL * (v21 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4))));
          if ( v20 == v24 )
            goto LABEL_9;
          v25 = 1;
          while ( v24 != -8 )
          {
            v23 = v21 & (v25 + v23);
            v24 = *(_QWORD *)(v22 + 8LL * v23);
            if ( v20 == v24 )
              goto LABEL_9;
            ++v25;
          }
        }
      }
      v26 = *(_DWORD *)(v11 + 24);
      if ( !v26 )
      {
        ++*(_QWORD *)v11;
LABEL_52:
        v58 = v11;
        sub_1353F00(v11, 2 * v26);
        v11 = v58;
        v45 = *(_DWORD *)(v58 + 24);
        if ( !v45 )
          goto LABEL_82;
        v16 = v67[0];
        v46 = v45 - 1;
        v47 = *(_QWORD *)(v58 + 8);
        v48 = (v45 - 1) & ((LODWORD(v67[0]) >> 9) ^ (LODWORD(v67[0]) >> 4));
        v31 = (char **)(v47 + 8LL * v48);
        v33 = *(_DWORD *)(v58 + 16) + 1;
        v49 = *v31;
        if ( *v31 != v67[0] )
        {
          v59 = 1;
          v50 = 0;
          while ( v49 != (char *)-8LL )
          {
            if ( !v50 && v49 == (char *)-16LL )
              v50 = v31;
            v48 = v46 & (v59 + v48);
            v31 = (char **)(v47 + 8LL * v48);
            v49 = *v31;
            if ( v67[0] == *v31 )
              goto LABEL_25;
            ++v59;
          }
          if ( v50 )
            v31 = v50;
        }
        goto LABEL_25;
      }
      v27 = *(_QWORD *)(v11 + 8);
      v28 = (v26 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v29 = (char **)(v27 + 8LL * v28);
      v30 = *v29;
      if ( v16 == *v29 )
      {
LABEL_9:
        v9 += 3;
        if ( v9 == v10 )
          goto LABEL_31;
      }
      else
      {
        v56 = 1;
        v31 = 0;
        while ( v30 != (char *)-8LL )
        {
          if ( v30 != (char *)-16LL || v31 )
            v29 = v31;
          v28 = (v26 - 1) & (v56 + v28);
          v30 = *(char **)(v27 + 8LL * v28);
          if ( v16 == v30 )
            goto LABEL_9;
          ++v56;
          v31 = v29;
          v29 = (char **)(v27 + 8LL * v28);
        }
        if ( !v31 )
          v31 = v29;
        v32 = *(_DWORD *)(v11 + 16);
        ++*(_QWORD *)v11;
        v33 = v32 + 1;
        if ( 4 * (v32 + 1) >= 3 * v26 )
          goto LABEL_52;
        if ( v26 - *(_DWORD *)(v11 + 20) - v33 <= v26 >> 3 )
        {
          v60 = v11;
          sub_1353F00(v11, v26);
          v11 = v60;
          v51 = *(_DWORD *)(v60 + 24);
          if ( !v51 )
          {
LABEL_82:
            ++*(_DWORD *)(a2 + 16);
            BUG();
          }
          v52 = *(_QWORD *)(v60 + 8);
          v53 = 1;
          v61 = v51 - 1;
          v54 = (v51 - 1) & ((LODWORD(v67[0]) >> 9) ^ (LODWORD(v67[0]) >> 4));
          v31 = (char **)(v52 + 8LL * v54);
          v33 = *(_DWORD *)(v11 + 16) + 1;
          v55 = 0;
          v16 = *v31;
          if ( v67[0] != *v31 )
          {
            while ( v16 != (char *)-8LL )
            {
              if ( !v55 && v16 == (char *)-16LL )
                v55 = v31;
              v54 = v61 & (v53 + v54);
              v31 = (char **)(v52 + 8LL * v54);
              v16 = *v31;
              if ( v67[0] == *v31 )
                goto LABEL_25;
              ++v53;
            }
            v16 = v67[0];
            if ( v55 )
              v31 = v55;
          }
        }
LABEL_25:
        *(_DWORD *)(v11 + 16) = v33;
        if ( *v31 != (char *)-8LL )
          --*(_DWORD *)(v11 + 20);
        *v31 = v16;
        v34 = *(char ***)(v11 + 40);
        if ( v34 == *(char ***)(v11 + 48) )
        {
          v57 = v11;
          sub_1287830(v11 + 32, v34, v67);
          v11 = v57;
          goto LABEL_9;
        }
        if ( v34 )
        {
          *v34 = v67[0];
          v34 = *(char ***)(v11 + 40);
        }
        v9 += 3;
        *(_QWORD *)(v11 + 40) = v34 + 1;
        if ( v9 == v10 )
          goto LABEL_31;
      }
    }
  }
  return result;
}
