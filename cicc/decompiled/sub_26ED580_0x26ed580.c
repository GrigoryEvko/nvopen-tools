// Function: sub_26ED580
// Address: 0x26ed580
//
void __fastcall sub_26ED580(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r14
  unsigned __int64 v5; // rax
  unsigned int v6; // esi
  __int64 v7; // rbx
  __int64 v8; // r8
  __int64 v9; // r11
  _QWORD *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rdx
  char v14; // cl
  __int64 v15; // rax
  int v16; // ecx
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  int v19; // r15d
  unsigned int v20; // esi
  __int64 v21; // r8
  unsigned int v22; // edi
  __int64 v23; // rcx
  int v24; // r11d
  _QWORD *v25; // r10
  int v26; // edi
  int v27; // ecx
  int v28; // r8d
  int v29; // r8d
  __int64 v30; // r9
  unsigned int v31; // edx
  __int64 v32; // rsi
  _QWORD *v33; // rdi
  int v34; // r8d
  int v35; // r8d
  __int64 v36; // r9
  unsigned int v37; // edx
  __int64 v38; // rsi
  int v39; // edx
  _QWORD *v40; // rdi
  int v41; // edi
  int v42; // edx
  int v43; // r9d
  int v44; // r9d
  __int64 v45; // r10
  unsigned int v46; // ecx
  __int64 v47; // r8
  int v48; // edi
  _QWORD *v49; // rsi
  int v50; // r8d
  int v51; // r8d
  __int64 v52; // r9
  int v53; // ecx
  __int64 v54; // r15
  _QWORD *v55; // rdi
  __int64 v56; // rsi
  unsigned int v57; // [rsp+Ch] [rbp-34h]

  v2 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  v3 = *(_QWORD *)a1 + 72LL;
  if ( v2 != v3 )
  {
    while ( 1 )
    {
LABEL_2:
      if ( !v2 )
        BUG();
      v5 = *(_QWORD *)(v2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v5 == v2 + 24 || !v5 || (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
        BUG();
      if ( *(_BYTE *)(v5 - 24) != 34 )
        goto LABEL_33;
      v6 = *(_DWORD *)(a2 + 24);
      v7 = *(_QWORD *)(v5 - 120);
      if ( !v6 )
      {
        ++*(_QWORD *)a2;
        goto LABEL_71;
      }
      v8 = *(_QWORD *)(a2 + 8);
      LODWORD(v9) = (v6 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v10 = (_QWORD *)(v8 + 8LL * (unsigned int)v9);
      v11 = *v10;
      if ( v7 != *v10 )
      {
        v39 = 1;
        v40 = 0;
        while ( v11 != -4096 )
        {
          if ( !v40 && v11 == -8192 )
            v40 = v10;
          v9 = (v6 - 1) & ((_DWORD)v9 + v39);
          v10 = (_QWORD *)(v8 + 8 * v9);
          v11 = *v10;
          if ( v7 == *v10 )
            goto LABEL_9;
          ++v39;
        }
        if ( v40 )
          v10 = v40;
        v41 = *(_DWORD *)(a2 + 16);
        ++*(_QWORD *)a2;
        v42 = v41 + 1;
        if ( 4 * (v41 + 1) < 3 * v6 )
        {
          if ( v6 - *(_DWORD *)(a2 + 20) - v42 <= v6 >> 3 )
          {
            sub_CF28B0(a2, v6);
            v50 = *(_DWORD *)(a2 + 24);
            if ( !v50 )
            {
LABEL_105:
              ++*(_DWORD *)(a2 + 16);
              BUG();
            }
            v51 = v50 - 1;
            v52 = *(_QWORD *)(a2 + 8);
            v53 = 1;
            LODWORD(v54) = v51 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
            v42 = *(_DWORD *)(a2 + 16) + 1;
            v55 = 0;
            v10 = (_QWORD *)(v52 + 8LL * (unsigned int)v54);
            v56 = *v10;
            if ( v7 != *v10 )
            {
              while ( v56 != -4096 )
              {
                if ( !v55 && v56 == -8192 )
                  v55 = v10;
                v54 = v51 & (unsigned int)(v54 + v53);
                v10 = (_QWORD *)(v52 + 8 * v54);
                v56 = *v10;
                if ( v7 == *v10 )
                  goto LABEL_63;
                ++v53;
              }
              if ( v55 )
                v10 = v55;
            }
          }
          goto LABEL_63;
        }
LABEL_71:
        sub_CF28B0(a2, 2 * v6);
        v43 = *(_DWORD *)(a2 + 24);
        if ( !v43 )
          goto LABEL_105;
        v44 = v43 - 1;
        v45 = *(_QWORD *)(a2 + 8);
        v46 = v44 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v42 = *(_DWORD *)(a2 + 16) + 1;
        v10 = (_QWORD *)(v45 + 8LL * v46);
        v47 = *v10;
        if ( v7 != *v10 )
        {
          v48 = 1;
          v49 = 0;
          while ( v47 != -4096 )
          {
            if ( v47 == -8192 && !v49 )
              v49 = v10;
            v46 = v44 & (v48 + v46);
            v10 = (_QWORD *)(v45 + 8LL * v46);
            v47 = *v10;
            if ( v7 == *v10 )
              goto LABEL_63;
            ++v48;
          }
          if ( v49 )
            v10 = v49;
        }
LABEL_63:
        *(_DWORD *)(a2 + 16) = v42;
        if ( *v10 == -4096 )
          goto LABEL_32;
LABEL_31:
        --*(_DWORD *)(a2 + 20);
        goto LABEL_32;
      }
LABEL_9:
      v12 = *(_QWORD *)(v7 + 16);
      if ( v12 )
        break;
LABEL_33:
      v2 = *(_QWORD *)(v2 + 8);
      if ( v3 == v2 )
        return;
    }
    while ( 1 )
    {
      v13 = v12;
      while ( 1 )
      {
        v14 = **(_BYTE **)(v13 + 24);
        v15 = v13;
        v13 = *(_QWORD *)(v13 + 8);
        if ( (unsigned __int8)(v14 - 30) <= 0xAu )
          break;
        if ( !v13 )
        {
          v2 = *(_QWORD *)(v2 + 8);
          if ( v3 == v2 )
            return;
          goto LABEL_2;
        }
      }
      v16 = 0;
      while ( 1 )
      {
        v15 = *(_QWORD *)(v15 + 8);
        if ( !v15 )
          break;
        while ( (unsigned __int8)(**(_BYTE **)(v15 + 24) - 30) <= 0xAu )
        {
          v15 = *(_QWORD *)(v15 + 8);
          ++v16;
          if ( !v15 )
            goto LABEL_16;
        }
      }
LABEL_16:
      if ( v16 )
        goto LABEL_33;
      while ( 1 )
      {
        v17 = *(_QWORD *)(v12 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v17 - 30) <= 0xAu )
          break;
        v12 = *(_QWORD *)(v12 + 8);
        if ( !v12 )
          BUG();
      }
      v7 = *(_QWORD *)(v17 + 40);
      v18 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v18 == v7 + 48 )
        goto LABEL_33;
      if ( !v18 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v18 - 24) - 30 > 0xA )
        goto LABEL_33;
      v19 = sub_B46E30(v18 - 24);
      if ( v19 != 1 )
        goto LABEL_33;
      v20 = *(_DWORD *)(a2 + 24);
      if ( v20 )
      {
        v21 = *(_QWORD *)(a2 + 8);
        v22 = (v20 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v10 = (_QWORD *)(v21 + 8LL * v22);
        v23 = *v10;
        if ( v7 == *v10 )
          goto LABEL_9;
        v24 = 1;
        v25 = 0;
        while ( v23 != -4096 )
        {
          if ( v23 == -8192 && !v25 )
            v25 = v10;
          v22 = (v20 - 1) & (v24 + v22);
          v10 = (_QWORD *)(v21 + 8LL * v22);
          v23 = *v10;
          if ( v7 == *v10 )
            goto LABEL_9;
          ++v24;
        }
        v26 = *(_DWORD *)(a2 + 16);
        if ( v25 )
          v10 = v25;
        ++*(_QWORD *)a2;
        v27 = v26 + 1;
        if ( 4 * (v26 + 1) < 3 * v20 )
        {
          if ( v20 - *(_DWORD *)(a2 + 20) - v27 <= v20 >> 3 )
          {
            v57 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
            sub_CF28B0(a2, v20);
            v34 = *(_DWORD *)(a2 + 24);
            if ( !v34 )
              goto LABEL_105;
            v35 = v34 - 1;
            v36 = *(_QWORD *)(a2 + 8);
            v37 = v35 & v57;
            v27 = *(_DWORD *)(a2 + 16) + 1;
            v33 = 0;
            v10 = (_QWORD *)(v36 + 8LL * (v35 & v57));
            v38 = *v10;
            if ( v7 != *v10 )
            {
              while ( v38 != -4096 )
              {
                if ( v38 == -8192 && !v33 )
                  v33 = v10;
                v37 = v35 & (v19 + v37);
                v10 = (_QWORD *)(v36 + 8LL * v37);
                v38 = *v10;
                if ( v7 == *v10 )
                  goto LABEL_30;
                ++v19;
              }
LABEL_54:
              if ( v33 )
                v10 = v33;
              goto LABEL_30;
            }
          }
          goto LABEL_30;
        }
      }
      else
      {
        ++*(_QWORD *)a2;
      }
      sub_CF28B0(a2, 2 * v20);
      v28 = *(_DWORD *)(a2 + 24);
      if ( !v28 )
        goto LABEL_105;
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a2 + 8);
      v31 = v29 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v27 = *(_DWORD *)(a2 + 16) + 1;
      v10 = (_QWORD *)(v30 + 8LL * v31);
      v32 = *v10;
      if ( v7 != *v10 )
      {
        v33 = 0;
        while ( v32 != -4096 )
        {
          if ( v32 == -8192 && !v33 )
            v33 = v10;
          v31 = v29 & (v19 + v31);
          v10 = (_QWORD *)(v30 + 8LL * v31);
          v32 = *v10;
          if ( v7 == *v10 )
            goto LABEL_30;
          ++v19;
        }
        goto LABEL_54;
      }
LABEL_30:
      *(_DWORD *)(a2 + 16) = v27;
      if ( *v10 != -4096 )
        goto LABEL_31;
LABEL_32:
      *v10 = v7;
      v12 = *(_QWORD *)(v7 + 16);
      if ( !v12 )
        goto LABEL_33;
    }
  }
}
