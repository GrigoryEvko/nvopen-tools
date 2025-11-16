// Function: sub_1548410
// Address: 0x1548410
//
__int64 __fastcall sub_1548410(__int64 a1, __int64 a2)
{
  __int64 v2; // r11
  unsigned int v4; // esi
  __int64 v5; // r8
  unsigned int v6; // edi
  __int64 *v7; // rax
  __int64 v8; // rcx
  int v9; // eax
  int v11; // r10d
  __int64 *v12; // rdx
  int v13; // eax
  int v14; // ecx
  __int64 v15; // rax
  int v16; // r14d
  __int64 v17; // r12
  __int64 v18; // r10
  __int64 v19; // rdi
  unsigned int v20; // ecx
  __int64 *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rbx
  int v24; // esi
  int v25; // esi
  __int64 v26; // r8
  unsigned int v27; // ecx
  int v28; // edx
  __int64 v29; // rdi
  int v30; // eax
  int v31; // eax
  __int64 v32; // r8
  unsigned int v33; // edi
  __int64 v34; // rsi
  int v35; // r10d
  __int64 *v36; // r9
  __int64 *v37; // r11
  int v38; // edx
  int v39; // ecx
  int v40; // ecx
  __int64 v41; // rdi
  unsigned int v42; // r15d
  __int64 v43; // rsi
  int v44; // r11d
  __int64 *v45; // r8
  int v46; // eax
  int v47; // eax
  __int64 v48; // r8
  int v49; // r10d
  unsigned int v50; // edi
  __int64 v51; // rsi
  int v52; // r11d
  __int64 *v53; // r15
  int v54; // eax
  int v55; // eax
  __int64 v56; // [rsp+10h] [rbp-50h]
  int v57; // [rsp+10h] [rbp-50h]
  __int64 v58; // [rsp+10h] [rbp-50h]
  __int64 v59; // [rsp+18h] [rbp-48h]
  __int64 v60; // [rsp+18h] [rbp-48h]
  __int64 v61; // [rsp+18h] [rbp-48h]
  __int64 v62; // [rsp+20h] [rbp-40h]
  unsigned int v63; // [rsp+2Ch] [rbp-34h]

  v2 = a2;
  v63 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v4 = *(_DWORD *)(a1 + 464);
  v62 = a1 + 440;
  while ( 2 )
  {
    if ( !v4 )
      goto LABEL_30;
LABEL_3:
    v5 = *(_QWORD *)(a1 + 448);
    v6 = (v4 - 1) & v63;
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( v2 != *v7 )
    {
      v11 = 1;
      v12 = 0;
      while ( v8 != -8 )
      {
        if ( !v12 && v8 == -16 )
          v12 = v7;
        v6 = (v4 - 1) & (v11 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( *v7 == v2 )
          goto LABEL_4;
        ++v11;
      }
      if ( !v12 )
        v12 = v7;
      v13 = *(_DWORD *)(a1 + 456);
      ++*(_QWORD *)(a1 + 440);
      v14 = v13 + 1;
      if ( 4 * (v13 + 1) >= 3 * v4 )
      {
LABEL_31:
        v60 = v2;
        sub_137BFC0(v62, 2 * v4);
        v30 = *(_DWORD *)(a1 + 464);
        if ( v30 )
        {
          v31 = v30 - 1;
          v2 = v60;
          v32 = *(_QWORD *)(a1 + 448);
          v33 = v31 & v63;
          v14 = *(_DWORD *)(a1 + 456) + 1;
          v12 = (__int64 *)(v32 + 16LL * (v31 & v63));
          v34 = *v12;
          if ( *v12 == v60 )
            goto LABEL_12;
          v35 = 1;
          v36 = 0;
          while ( v34 != -8 )
          {
            if ( !v36 && v34 == -16 )
              v36 = v12;
            v33 = v31 & (v35 + v33);
            v12 = (__int64 *)(v32 + 16LL * v33);
            v34 = *v12;
            if ( *v12 == v60 )
              goto LABEL_12;
            ++v35;
          }
LABEL_35:
          if ( v36 )
            v12 = v36;
          goto LABEL_12;
        }
      }
      else
      {
        if ( v4 - *(_DWORD *)(a1 + 460) - v14 > v4 >> 3 )
        {
LABEL_12:
          *(_DWORD *)(a1 + 456) = v14;
          if ( *v12 != -8 )
            --*(_DWORD *)(a1 + 460);
          *v12 = v2;
          *((_DWORD *)v12 + 2) = 0;
          v4 = *(_DWORD *)(a1 + 464);
          goto LABEL_15;
        }
        v61 = v2;
        sub_137BFC0(v62, v4);
        v46 = *(_DWORD *)(a1 + 464);
        if ( v46 )
        {
          v47 = v46 - 1;
          v48 = *(_QWORD *)(a1 + 448);
          v36 = 0;
          v2 = v61;
          v49 = 1;
          v50 = v47 & v63;
          v14 = *(_DWORD *)(a1 + 456) + 1;
          v12 = (__int64 *)(v48 + 16LL * (v47 & v63));
          v51 = *v12;
          if ( *v12 == v61 )
            goto LABEL_12;
          while ( v51 != -8 )
          {
            if ( !v36 && v51 == -16 )
              v36 = v12;
            v50 = v47 & (v49 + v50);
            v12 = (__int64 *)(v48 + 16LL * v50);
            v51 = *v12;
            if ( *v12 == v61 )
              goto LABEL_12;
            ++v49;
          }
          goto LABEL_35;
        }
      }
      ++*(_DWORD *)(a1 + 456);
      BUG();
    }
LABEL_4:
    v9 = *((_DWORD *)v7 + 2);
    if ( !v9 )
    {
LABEL_15:
      v15 = *(_QWORD *)(v2 + 56);
      v16 = 0;
      v17 = *(_QWORD *)(v15 + 80);
      v18 = v15 + 72;
      if ( v17 == v15 + 72 )
        continue;
      v59 = v2;
      while ( 1 )
      {
        v23 = v17 - 24;
        if ( !v17 )
          v23 = 0;
        ++v16;
        if ( !v4 )
          break;
        v19 = *(_QWORD *)(a1 + 448);
        v20 = (v4 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v21 = (__int64 *)(v19 + 16LL * v20);
        v22 = *v21;
        if ( v23 == *v21 )
        {
LABEL_18:
          *((_DWORD *)v21 + 2) = v16;
          v17 = *(_QWORD *)(v17 + 8);
          if ( v18 == v17 )
            goto LABEL_29;
          goto LABEL_19;
        }
        v57 = 1;
        v37 = 0;
        while ( v22 != -8 )
        {
          if ( !v37 && v22 == -16 )
            v37 = v21;
          v20 = (v4 - 1) & (v57 + v20);
          v21 = (__int64 *)(v19 + 16LL * v20);
          v22 = *v21;
          if ( v23 == *v21 )
            goto LABEL_18;
          ++v57;
        }
        v38 = *(_DWORD *)(a1 + 456);
        if ( v37 )
          v21 = v37;
        ++*(_QWORD *)(a1 + 440);
        v28 = v38 + 1;
        if ( 4 * v28 >= 3 * v4 )
          goto LABEL_24;
        if ( v4 - *(_DWORD *)(a1 + 460) - v28 <= v4 >> 3 )
        {
          v58 = v18;
          sub_137BFC0(v62, v4);
          v39 = *(_DWORD *)(a1 + 464);
          if ( !v39 )
          {
LABEL_90:
            ++*(_DWORD *)(a1 + 456);
            BUG();
          }
          v40 = v39 - 1;
          v41 = *(_QWORD *)(a1 + 448);
          v18 = v58;
          v42 = v40 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v28 = *(_DWORD *)(a1 + 456) + 1;
          v21 = (__int64 *)(v41 + 16LL * v42);
          v43 = *v21;
          if ( *v21 != v23 )
          {
            v44 = 1;
            v45 = 0;
            while ( v43 != -8 )
            {
              if ( !v45 && v43 == -16 )
                v45 = v21;
              v54 = v44++;
              v42 = v40 & (v54 + v42);
              v21 = (__int64 *)(v41 + 16LL * v42);
              v43 = *v21;
              if ( v23 == *v21 )
                goto LABEL_26;
            }
            if ( v45 )
              v21 = v45;
          }
        }
LABEL_26:
        *(_DWORD *)(a1 + 456) = v28;
        if ( *v21 != -8 )
          --*(_DWORD *)(a1 + 460);
        *((_DWORD *)v21 + 2) = 0;
        *v21 = v23;
        *((_DWORD *)v21 + 2) = v16;
        v17 = *(_QWORD *)(v17 + 8);
        if ( v18 == v17 )
        {
LABEL_29:
          v4 = *(_DWORD *)(a1 + 464);
          v2 = v59;
          if ( !v4 )
          {
LABEL_30:
            ++*(_QWORD *)(a1 + 440);
            goto LABEL_31;
          }
          goto LABEL_3;
        }
LABEL_19:
        v4 = *(_DWORD *)(a1 + 464);
      }
      ++*(_QWORD *)(a1 + 440);
LABEL_24:
      v56 = v18;
      sub_137BFC0(v62, 2 * v4);
      v24 = *(_DWORD *)(a1 + 464);
      if ( !v24 )
        goto LABEL_90;
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 448);
      v18 = v56;
      v27 = v25 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v28 = *(_DWORD *)(a1 + 456) + 1;
      v21 = (__int64 *)(v26 + 16LL * v27);
      v29 = *v21;
      if ( *v21 != v23 )
      {
        v52 = 1;
        v53 = 0;
        while ( v29 != -8 )
        {
          if ( !v53 && v29 == -16 )
            v53 = v21;
          v55 = v52++;
          v27 = v25 & (v55 + v27);
          v21 = (__int64 *)(v26 + 16LL * v27);
          v29 = *v21;
          if ( v23 == *v21 )
            goto LABEL_26;
        }
        if ( v53 )
          v21 = v53;
      }
      goto LABEL_26;
    }
    return (unsigned int)(v9 - 1);
  }
}
