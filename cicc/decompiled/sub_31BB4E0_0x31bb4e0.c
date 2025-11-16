// Function: sub_31BB4E0
// Address: 0x31bb4e0
//
void __fastcall sub_31BB4E0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r12
  unsigned int v8; // esi
  __int64 v9; // r8
  unsigned int v10; // edi
  _QWORD *v11; // rcx
  __int64 v12; // rdx
  unsigned int v13; // esi
  __int64 v14; // r9
  __int64 v15; // rdi
  unsigned int v16; // ecx
  _QWORD *v17; // rdx
  __int64 v18; // rax
  int v19; // r11d
  _QWORD *v20; // r10
  int v21; // ecx
  int v22; // edx
  int v23; // edx
  int v24; // edx
  __int64 v25; // rdi
  __int64 v26; // rcx
  _QWORD *v27; // r10
  __int64 v28; // rsi
  int v29; // eax
  int v30; // r11d
  _QWORD *v31; // r9
  int v32; // r11d
  int v33; // eax
  int v34; // ecx
  int v35; // ecx
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rsi
  int v39; // r11d
  _QWORD *v40; // r9
  int v41; // edx
  int v42; // edx
  __int64 v43; // rdi
  int v44; // r11d
  __int64 v45; // rcx
  __int64 v46; // rsi
  int v47; // ecx
  int v48; // ecx
  __int64 v49; // rdi
  int v50; // r11d
  __int64 v51; // rax
  __int64 v52; // rsi
  unsigned int v53; // [rsp+8h] [rbp-58h]
  __int64 v54; // [rsp+10h] [rbp-50h]
  unsigned int v55; // [rsp+1Ch] [rbp-44h]
  __int64 v57; // [rsp+28h] [rbp-38h]

  v5 = a3[1];
  v6 = *(_QWORD *)(a2 + 8);
  v57 = *a3;
  if ( v5 )
    v5 = *(_QWORD *)(v5 + 48);
  if ( v57 != v5 )
  {
    v55 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    v54 = a2 + 56;
    do
    {
      while ( v5 )
      {
        v7 = *(_QWORD *)(v5 + 40);
        if ( !sub_31B9400(a1, *(_QWORD *)(v7 + 8), v6) )
          goto LABEL_6;
LABEL_11:
        v8 = *(_DWORD *)(a2 + 80);
        if ( v8 )
        {
          v9 = *(_QWORD *)(a2 + 64);
          v10 = (v8 - 1) & (((unsigned int)v7 >> 4) ^ ((unsigned int)v7 >> 9));
          v11 = (_QWORD *)(v9 + 8LL * v10);
          v12 = *v11;
          if ( *v11 == v7 )
          {
LABEL_13:
            v13 = *(_DWORD *)(v7 + 112);
            v14 = v7 + 88;
            if ( !v13 )
              goto LABEL_28;
            goto LABEL_14;
          }
          v19 = 1;
          v20 = 0;
          while ( v12 != -4096 )
          {
            if ( v12 != -8192 || v20 )
              v11 = v20;
            v10 = (v8 - 1) & (v19 + v10);
            v12 = *(_QWORD *)(v9 + 8LL * v10);
            if ( v12 == v7 )
              goto LABEL_13;
            ++v19;
            v20 = v11;
            v11 = (_QWORD *)(v9 + 8LL * v10);
          }
          if ( !v20 )
            v20 = v11;
          v21 = *(_DWORD *)(a2 + 72);
          ++*(_QWORD *)(a2 + 56);
          v22 = v21 + 1;
          if ( 4 * (v21 + 1) < 3 * v8 )
          {
            if ( v8 - *(_DWORD *)(a2 + 76) - v22 <= v8 >> 3 )
            {
              v53 = ((unsigned int)v7 >> 4) ^ ((unsigned int)v7 >> 9);
              sub_31BB310(v54, v8);
              v47 = *(_DWORD *)(a2 + 80);
              if ( !v47 )
              {
LABEL_89:
                ++*(_DWORD *)(a2 + 72);
                BUG();
              }
              v48 = v47 - 1;
              v49 = *(_QWORD *)(a2 + 64);
              v50 = 1;
              v40 = 0;
              LODWORD(v51) = v48 & v53;
              v20 = (_QWORD *)(v49 + 8LL * (v48 & v53));
              v52 = *v20;
              v22 = *(_DWORD *)(a2 + 72) + 1;
              if ( v7 != *v20 )
              {
                while ( v52 != -4096 )
                {
                  if ( !v40 && v52 == -8192 )
                    v40 = v20;
                  v51 = v48 & (unsigned int)(v51 + v50);
                  v20 = (_QWORD *)(v49 + 8 * v51);
                  v52 = *v20;
                  if ( *v20 == v7 )
                    goto LABEL_25;
                  ++v50;
                }
LABEL_66:
                if ( v40 )
                  v20 = v40;
                goto LABEL_25;
              }
            }
            goto LABEL_25;
          }
        }
        else
        {
          ++*(_QWORD *)(a2 + 56);
        }
        sub_31BB310(v54, 2 * v8);
        v34 = *(_DWORD *)(a2 + 80);
        if ( !v34 )
          goto LABEL_89;
        v35 = v34 - 1;
        v36 = *(_QWORD *)(a2 + 64);
        LODWORD(v37) = v35 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v20 = (_QWORD *)(v36 + 8LL * (unsigned int)v37);
        v38 = *v20;
        v22 = *(_DWORD *)(a2 + 72) + 1;
        if ( v7 != *v20 )
        {
          v39 = 1;
          v40 = 0;
          while ( v38 != -4096 )
          {
            if ( v38 == -8192 && !v40 )
              v40 = v20;
            v37 = v35 & (unsigned int)(v37 + v39);
            v20 = (_QWORD *)(v36 + 8 * v37);
            v38 = *v20;
            if ( *v20 == v7 )
              goto LABEL_25;
            ++v39;
          }
          goto LABEL_66;
        }
LABEL_25:
        *(_DWORD *)(a2 + 72) = v22;
        if ( *v20 != -4096 )
          --*(_DWORD *)(a2 + 76);
        *v20 = v7;
        v13 = *(_DWORD *)(v7 + 112);
        v14 = v7 + 88;
        if ( !v13 )
        {
LABEL_28:
          ++*(_QWORD *)(v7 + 88);
          goto LABEL_29;
        }
LABEL_14:
        v15 = *(_QWORD *)(v7 + 96);
        v16 = (v13 - 1) & v55;
        v17 = (_QWORD *)(v15 + 8LL * v16);
        v18 = *v17;
        if ( a2 == *v17 )
          goto LABEL_15;
        v32 = 1;
        v27 = 0;
        while ( v18 != -4096 )
        {
          if ( v27 || v18 != -8192 )
            v17 = v27;
          v16 = (v13 - 1) & (v32 + v16);
          v18 = *(_QWORD *)(v15 + 8LL * v16);
          if ( a2 == v18 )
            goto LABEL_15;
          ++v32;
          v27 = v17;
          v17 = (_QWORD *)(v15 + 8LL * v16);
        }
        v33 = *(_DWORD *)(v7 + 104);
        if ( !v27 )
          v27 = v17;
        ++*(_QWORD *)(v7 + 88);
        v29 = v33 + 1;
        if ( 4 * v29 < 3 * v13 )
        {
          if ( v13 - *(_DWORD *)(v7 + 108) - v29 > v13 >> 3 )
            goto LABEL_44;
          sub_31BB310(v14, v13);
          v41 = *(_DWORD *)(v7 + 112);
          if ( !v41 )
          {
LABEL_90:
            ++*(_DWORD *)(v7 + 104);
            BUG();
          }
          v42 = v41 - 1;
          v43 = *(_QWORD *)(v7 + 96);
          v31 = 0;
          v44 = 1;
          LODWORD(v45) = v42 & v55;
          v27 = (_QWORD *)(v43 + 8LL * (v42 & v55));
          v46 = *v27;
          v29 = *(_DWORD *)(v7 + 104) + 1;
          if ( a2 == *v27 )
            goto LABEL_44;
          while ( v46 != -4096 )
          {
            if ( v46 == -8192 && !v31 )
              v31 = v27;
            v45 = v42 & (unsigned int)(v45 + v44);
            v27 = (_QWORD *)(v43 + 8 * v45);
            v46 = *v27;
            if ( a2 == *v27 )
              goto LABEL_44;
            ++v44;
          }
          goto LABEL_60;
        }
LABEL_29:
        sub_31BB310(v14, 2 * v13);
        v23 = *(_DWORD *)(v7 + 112);
        if ( !v23 )
          goto LABEL_90;
        v24 = v23 - 1;
        v25 = *(_QWORD *)(v7 + 96);
        LODWORD(v26) = v24 & v55;
        v27 = (_QWORD *)(v25 + 8LL * (v24 & v55));
        v28 = *v27;
        v29 = *(_DWORD *)(v7 + 104) + 1;
        if ( a2 == *v27 )
          goto LABEL_44;
        v30 = 1;
        v31 = 0;
        while ( v28 != -4096 )
        {
          if ( !v31 && v28 == -8192 )
            v31 = v27;
          v26 = v24 & (unsigned int)(v26 + v30);
          v27 = (_QWORD *)(v25 + 8 * v26);
          v28 = *v27;
          if ( a2 == *v27 )
            goto LABEL_44;
          ++v30;
        }
LABEL_60:
        if ( v31 )
          v27 = v31;
LABEL_44:
        *(_DWORD *)(v7 + 104) = v29;
        if ( *v27 != -4096 )
          --*(_DWORD *)(v7 + 108);
        *v27 = a2;
LABEL_15:
        if ( !*(_BYTE *)(a2 + 24) )
          ++*(_DWORD *)(v7 + 20);
        if ( !v5 )
          goto LABEL_9;
LABEL_6:
        v5 = *(_QWORD *)(v5 + 40);
        if ( v57 == v5 )
          return;
      }
      v7 = a3[1];
      if ( sub_31B9400(a1, *(_QWORD *)(v7 + 8), v6) )
        goto LABEL_11;
LABEL_9:
      v5 = a3[1];
    }
    while ( v57 != v5 );
  }
}
