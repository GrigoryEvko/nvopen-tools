// Function: sub_1BC8DF0
// Address: 0x1bc8df0
//
__int64 __fastcall sub_1BC8DF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r12
  __int64 v9; // r8
  unsigned int v10; // edi
  __int64 *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rbx
  int v14; // eax
  __int64 v15; // rdx
  unsigned int v16; // esi
  int v17; // eax
  int v18; // edi
  __int64 v19; // rsi
  __int64 v20; // rax
  int v21; // ecx
  __int64 *v22; // rdx
  __int64 v23; // r8
  __int64 v24; // rax
  unsigned int v25; // esi
  __int64 v26; // r8
  unsigned int v27; // ecx
  __int64 *v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 result; // rax
  int v32; // r10d
  int v33; // eax
  int v34; // eax
  int v35; // eax
  __int64 v36; // rdi
  __int64 *v37; // r8
  __int64 v38; // rbx
  int v39; // r9d
  __int64 v40; // rsi
  int v41; // r11d
  __int64 *v42; // r10
  int v43; // ecx
  int v44; // ecx
  int v45; // r9d
  int v46; // r9d
  __int64 v47; // r10
  unsigned int v48; // edx
  __int64 v49; // r8
  int v50; // edi
  __int64 *v51; // rsi
  int v52; // r8d
  int v53; // r8d
  __int64 v54; // r10
  int v55; // edi
  unsigned int v56; // edx
  __int64 v57; // r9
  int v58; // r10d
  __int64 *v59; // r9
  unsigned int v60; // [rsp+Ch] [rbp-44h]
  __int64 v61; // [rsp+10h] [rbp-40h]

  if ( a2 != a3 )
  {
    v7 = a2;
    v61 = a1 + 40;
    while ( 1 )
    {
      v16 = *(_DWORD *)(a1 + 64);
      if ( !v16 )
        break;
      v9 = *(_QWORD *)(a1 + 48);
      v10 = (v16 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v11 = (__int64 *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( v7 != *v11 )
      {
        v32 = 1;
        v22 = 0;
        while ( v12 != -8 )
        {
          if ( v12 == -16 && !v22 )
            v22 = v11;
          v10 = (v16 - 1) & (v32 + v10);
          v11 = (__int64 *)(v9 + 16LL * v10);
          v12 = *v11;
          if ( *v11 == v7 )
            goto LABEL_4;
          ++v32;
        }
        if ( !v22 )
          v22 = v11;
        v33 = *(_DWORD *)(a1 + 56);
        ++*(_QWORD *)(a1 + 40);
        v21 = v33 + 1;
        if ( 4 * (v33 + 1) < 3 * v16 )
        {
          if ( v16 - *(_DWORD *)(a1 + 60) - v21 <= v16 >> 3 )
          {
            sub_1BC8C30(v61, v16);
            v34 = *(_DWORD *)(a1 + 64);
            if ( !v34 )
            {
LABEL_102:
              ++*(_DWORD *)(a1 + 56);
              BUG();
            }
            v35 = v34 - 1;
            v36 = *(_QWORD *)(a1 + 48);
            v37 = 0;
            LODWORD(v38) = v35 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
            v39 = 1;
            v21 = *(_DWORD *)(a1 + 56) + 1;
            v22 = (__int64 *)(v36 + 16LL * (unsigned int)v38);
            v40 = *v22;
            if ( v7 != *v22 )
            {
              while ( v40 != -8 )
              {
                if ( v40 == -16 && !v37 )
                  v37 = v22;
                v38 = v35 & (unsigned int)(v38 + v39);
                v22 = (__int64 *)(v36 + 16 * v38);
                v40 = *v22;
                if ( v7 == *v22 )
                  goto LABEL_16;
                ++v39;
              }
              if ( v37 )
                v22 = v37;
            }
          }
          goto LABEL_16;
        }
LABEL_14:
        sub_1BC8C30(v61, 2 * v16);
        v17 = *(_DWORD *)(a1 + 64);
        if ( !v17 )
          goto LABEL_102;
        v18 = v17 - 1;
        v19 = *(_QWORD *)(a1 + 48);
        LODWORD(v20) = (v17 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v21 = *(_DWORD *)(a1 + 56) + 1;
        v22 = (__int64 *)(v19 + 16LL * (unsigned int)v20);
        v23 = *v22;
        if ( v7 != *v22 )
        {
          v58 = 1;
          v59 = 0;
          while ( v23 != -8 )
          {
            if ( !v59 && v23 == -16 )
              v59 = v22;
            v20 = v18 & (unsigned int)(v20 + v58);
            v22 = (__int64 *)(v19 + 16 * v20);
            v23 = *v22;
            if ( v7 == *v22 )
              goto LABEL_16;
            ++v58;
          }
          if ( v59 )
            v22 = v59;
        }
LABEL_16:
        *(_DWORD *)(a1 + 56) = v21;
        if ( *v22 != -8 )
          --*(_DWORD *)(a1 + 60);
        *v22 = v7;
        v22[1] = 0;
        goto LABEL_19;
      }
LABEL_4:
      v13 = v11[1];
      if ( v13 )
        goto LABEL_5;
LABEL_19:
      v24 = sub_1BC27C0(a1);
      v25 = *(_DWORD *)(a1 + 64);
      v13 = v24;
      if ( !v25 )
      {
        ++*(_QWORD *)(a1 + 40);
        goto LABEL_62;
      }
      v26 = *(_QWORD *)(a1 + 48);
      v27 = (v25 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v28 = (__int64 *)(v26 + 16LL * v27);
      v29 = *v28;
      if ( v7 != *v28 )
      {
        v41 = 1;
        v42 = 0;
        while ( v29 != -8 )
        {
          if ( !v42 && v29 == -16 )
            v42 = v28;
          v27 = (v25 - 1) & (v41 + v27);
          v28 = (__int64 *)(v26 + 16LL * v27);
          v29 = *v28;
          if ( v7 == *v28 )
            goto LABEL_21;
          ++v41;
        }
        v43 = *(_DWORD *)(a1 + 56);
        if ( v42 )
          v28 = v42;
        ++*(_QWORD *)(a1 + 40);
        v44 = v43 + 1;
        if ( 4 * v44 < 3 * v25 )
        {
          if ( v25 - *(_DWORD *)(a1 + 60) - v44 <= v25 >> 3 )
          {
            v60 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
            sub_1BC8C30(v61, v25);
            v52 = *(_DWORD *)(a1 + 64);
            if ( !v52 )
            {
LABEL_103:
              ++*(_DWORD *)(a1 + 56);
              BUG();
            }
            v53 = v52 - 1;
            v54 = *(_QWORD *)(a1 + 48);
            v55 = 1;
            v51 = 0;
            v56 = v53 & v60;
            v44 = *(_DWORD *)(a1 + 56) + 1;
            v28 = (__int64 *)(v54 + 16LL * (v53 & v60));
            v57 = *v28;
            if ( v7 != *v28 )
            {
              while ( v57 != -8 )
              {
                if ( v57 == -16 && !v51 )
                  v51 = v28;
                v56 = v53 & (v55 + v56);
                v28 = (__int64 *)(v54 + 16LL * v56);
                v57 = *v28;
                if ( v7 == *v28 )
                  goto LABEL_58;
                ++v55;
              }
              goto LABEL_74;
            }
          }
          goto LABEL_58;
        }
LABEL_62:
        sub_1BC8C30(v61, 2 * v25);
        v45 = *(_DWORD *)(a1 + 64);
        if ( !v45 )
          goto LABEL_103;
        v46 = v45 - 1;
        v47 = *(_QWORD *)(a1 + 48);
        v48 = v46 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v44 = *(_DWORD *)(a1 + 56) + 1;
        v28 = (__int64 *)(v47 + 16LL * v48);
        v49 = *v28;
        if ( v7 != *v28 )
        {
          v50 = 1;
          v51 = 0;
          while ( v49 != -8 )
          {
            if ( !v51 && v49 == -16 )
              v51 = v28;
            v48 = v46 & (v50 + v48);
            v28 = (__int64 *)(v47 + 16LL * v48);
            v49 = *v28;
            if ( v7 == *v28 )
              goto LABEL_58;
            ++v50;
          }
LABEL_74:
          if ( v51 )
            v28 = v51;
        }
LABEL_58:
        *(_DWORD *)(a1 + 56) = v44;
        if ( *v28 != -8 )
          --*(_DWORD *)(a1 + 60);
        *v28 = v7;
        v28[1] = 0;
      }
LABEL_21:
      v28[1] = v13;
      *(_QWORD *)v13 = v7;
LABEL_5:
      v14 = *(_DWORD *)(a1 + 224);
      *(_QWORD *)(v13 + 8) = v13;
      *(_QWORD *)(v13 + 16) = 0;
      *(_QWORD *)(v13 + 24) = 0;
      *(_DWORD *)(v13 + 80) = v14;
      *(_QWORD *)(v13 + 88) = -1;
      *(_DWORD *)(v13 + 96) = -1;
      *(_BYTE *)(v13 + 100) = 0;
      *(_DWORD *)(v13 + 40) = 0;
      *(_QWORD *)(v13 + 104) = v7;
      if ( (unsigned __int8)sub_15F2ED0(v7) )
      {
        if ( *(_BYTE *)(v7 + 16) != 78 )
          goto LABEL_7;
      }
      else
      {
        if ( !(unsigned __int8)sub_15F3040(v7) )
          goto LABEL_9;
        if ( *(_BYTE *)(v7 + 16) != 78 )
        {
LABEL_7:
          if ( a4 )
            goto LABEL_8;
          goto LABEL_28;
        }
      }
      v30 = *(_QWORD *)(v7 - 24);
      if ( *(_BYTE *)(v30 + 16) || (*(_BYTE *)(v30 + 33) & 0x20) == 0 )
        goto LABEL_7;
      if ( *(_DWORD *)(v30 + 36) == 191 )
        goto LABEL_9;
      if ( a4 )
      {
LABEL_8:
        *(_QWORD *)(a4 + 24) = v13;
        a4 = v13;
        goto LABEL_9;
      }
LABEL_28:
      *(_QWORD *)(a1 + 200) = v13;
      a4 = v13;
LABEL_9:
      v15 = *(_QWORD *)(v7 + 32);
      if ( v15 == *(_QWORD *)(v7 + 40) + 40LL || !v15 )
      {
        v7 = 0;
        if ( !a3 )
          goto LABEL_30;
      }
      else
      {
        v7 = v15 - 24;
        if ( a3 == v15 - 24 )
          goto LABEL_30;
      }
    }
    ++*(_QWORD *)(a1 + 40);
    goto LABEL_14;
  }
LABEL_30:
  result = a5;
  if ( a5 )
  {
    if ( a4 )
      *(_QWORD *)(a4 + 24) = a5;
  }
  else
  {
    *(_QWORD *)(a1 + 208) = a4;
  }
  return result;
}
