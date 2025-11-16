// Function: sub_20FBA30
// Address: 0x20fba30
//
__int64 __fastcall sub_20FBA30(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 v10; // r8
  unsigned int v11; // esi
  __int64 v12; // rax
  _QWORD *v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rax
  _QWORD *v21; // rax
  unsigned int v22; // esi
  __int64 v23; // rdi
  unsigned int v24; // eax
  _QWORD *v25; // rbx
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  _QWORD *v29; // rdx
  int v30; // eax
  int v31; // eax
  int v32; // r10d
  __int64 v33; // rdx
  unsigned int v34; // esi
  int v35; // r11d
  _QWORD *v36; // rdi
  int v37; // eax
  int v38; // r10d
  __int64 v39; // rdx
  int v40; // r11d
  unsigned int v41; // esi
  int v42; // r10d
  int v43; // r10d
  __int64 v44; // r8
  unsigned int v45; // edx
  int v46; // eax
  __int64 v47; // rdi
  int v48; // r10d
  _QWORD *v49; // rcx
  int v50; // r9d
  int v51; // r9d
  __int64 v52; // rdi
  int v53; // ecx
  unsigned int v54; // r12d
  _QWORD *v55; // rdx
  __int64 v56; // rsi
  int v57; // esi
  _QWORD *v58; // rcx
  __int64 v59; // [rsp+0h] [rbp-70h]
  __int64 v62; // [rsp+18h] [rbp-58h]
  __int64 v63; // [rsp+20h] [rbp-50h]
  __int64 v64; // [rsp+20h] [rbp-50h]
  __int64 v65; // [rsp+20h] [rbp-50h]
  _QWORD *v66; // [rsp+28h] [rbp-48h]
  __int64 v67; // [rsp+28h] [rbp-48h]
  int v68; // [rsp+28h] [rbp-48h]

  result = *a1 + 320LL;
  v59 = result;
  v62 = *(_QWORD *)(*a1 + 328LL);
  if ( v62 != result )
  {
    while ( 1 )
    {
      v4 = 0;
      v5 = 0;
      v6 = 0;
      v7 = *(_QWORD *)(v62 + 32);
      if ( v62 + 24 == v7 )
        goto LABEL_12;
      do
      {
        v8 = sub_15C70A0(v7 + 64);
        v10 = v8;
        if ( !v8 || v8 == v6 )
        {
          v5 = v7;
LABEL_7:
          if ( !v7 )
            BUG();
        }
        else
        {
          switch ( **(_WORD **)(v7 + 16) )
          {
            case 2:
            case 3:
            case 4:
            case 6:
            case 9:
            case 0xC:
            case 0xD:
            case 0x11:
            case 0x12:
              goto LABEL_7;
            default:
              if ( !v4 )
                goto LABEL_27;
              v11 = *(_DWORD *)(a3 + 24);
              if ( v11 )
              {
                v9 = *(_QWORD *)(a3 + 8);
                v12 = (v11 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
                v13 = (_QWORD *)(v9 + 16 * v12);
                v14 = *v13;
                if ( v4 == *v13 )
                  goto LABEL_20;
                v68 = 1;
                v29 = 0;
                while ( v14 != -8 )
                {
                  if ( !v29 && v14 == -16 )
                    v29 = v13;
                  LODWORD(v12) = (v11 - 1) & (v68 + v12);
                  v13 = (_QWORD *)(v9 + 16LL * (unsigned int)v12);
                  v14 = *v13;
                  if ( *v13 == v4 )
                    goto LABEL_20;
                  ++v68;
                }
                if ( v29 )
                  v13 = v29;
                ++*(_QWORD *)a3;
                v30 = *(_DWORD *)(a3 + 16) + 1;
                if ( 4 * v30 < 3 * v11 )
                {
                  LODWORD(v9) = v11 >> 3;
                  if ( v11 - *(_DWORD *)(a3 + 20) - v30 <= v11 >> 3 )
                  {
                    v65 = v10;
                    sub_20FB870(a3, v11);
                    v37 = *(_DWORD *)(a3 + 24);
                    if ( !v37 )
                    {
LABEL_110:
                      ++*(_DWORD *)(a3 + 16);
                      BUG();
                    }
                    v38 = v37 - 1;
                    v39 = *(_QWORD *)(a3 + 8);
                    v36 = 0;
                    v10 = v65;
                    v40 = 1;
                    v41 = (v37 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
                    v30 = *(_DWORD *)(a3 + 16) + 1;
                    v13 = (_QWORD *)(v39 + 16LL * v41);
                    v9 = *v13;
                    if ( *v13 != v4 )
                    {
                      while ( v9 != -8 )
                      {
                        if ( !v36 && v9 == -16 )
                          v36 = v13;
                        v41 = v38 & (v40 + v41);
                        v13 = (_QWORD *)(v39 + 16LL * v41);
                        v9 = *v13;
                        if ( *v13 == v4 )
                          goto LABEL_41;
                        ++v40;
                      }
LABEL_49:
                      if ( v36 )
                        v13 = v36;
                      goto LABEL_41;
                    }
                  }
                  goto LABEL_41;
                }
              }
              else
              {
                ++*(_QWORD *)a3;
              }
              v64 = v10;
              sub_20FB870(a3, 2 * v11);
              v31 = *(_DWORD *)(a3 + 24);
              if ( !v31 )
                goto LABEL_110;
              v32 = v31 - 1;
              v33 = *(_QWORD *)(a3 + 8);
              v10 = v64;
              v34 = (v31 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
              v30 = *(_DWORD *)(a3 + 16) + 1;
              v13 = (_QWORD *)(v33 + 16LL * v34);
              v9 = *v13;
              if ( *v13 != v4 )
              {
                v35 = 1;
                v36 = 0;
                while ( v9 != -8 )
                {
                  if ( !v36 && v9 == -16 )
                    v36 = v13;
                  v34 = v32 & (v35 + v34);
                  v13 = (_QWORD *)(v33 + 16LL * v34);
                  v9 = *v13;
                  if ( *v13 == v4 )
                    goto LABEL_41;
                  ++v35;
                }
                goto LABEL_49;
              }
LABEL_41:
              *(_DWORD *)(a3 + 16) = v30;
              if ( *v13 != -8 )
                --*(_DWORD *)(a3 + 20);
              *v13 = v4;
              v13[1] = 0;
LABEL_20:
              if ( v6 )
              {
                v15 = *(unsigned int *)(v6 + 8);
                v16 = 0;
                if ( (_DWORD)v15 == 2 )
                  v16 = *(_QWORD *)(v6 - 8);
                v63 = v10;
                v66 = v13;
                v17 = sub_20FB390(a1, *(unsigned __int8 **)(v6 - 8 * v15), v16);
                v10 = v63;
                v13 = v66;
                v6 = v17;
              }
              v13[1] = v6;
              v18 = *(unsigned int *)(a2 + 8);
              if ( (unsigned int)v18 >= *(_DWORD *)(a2 + 12) )
              {
                v67 = v10;
                sub_16CD150(a2, (const void *)(a2 + 16), 0, 16, v10, v9);
                v18 = *(unsigned int *)(a2 + 8);
                v10 = v67;
              }
              v19 = (_QWORD *)(*(_QWORD *)a2 + 16 * v18);
              *v19 = v4;
              v19[1] = v5;
              ++*(_DWORD *)(a2 + 8);
LABEL_27:
              v4 = v7;
              v5 = v7;
              v6 = v10;
              break;
          }
        }
        if ( (*(_BYTE *)v7 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v7 + 46) & 8) != 0 )
            v7 = *(_QWORD *)(v7 + 8);
        }
        v7 = *(_QWORD *)(v7 + 8);
      }
      while ( v62 + 24 != v7 );
      if ( v4 == 0 || v5 == 0 || !v6 )
        goto LABEL_12;
      v20 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v20 >= *(_DWORD *)(a2 + 12) )
      {
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 16, v10, v9);
        v20 = *(unsigned int *)(a2 + 8);
      }
      v21 = (_QWORD *)(*(_QWORD *)a2 + 16 * v20);
      *v21 = v4;
      v21[1] = v5;
      ++*(_DWORD *)(a2 + 8);
      v22 = *(_DWORD *)(a3 + 24);
      if ( !v22 )
        break;
      v23 = *(_QWORD *)(a3 + 8);
      v24 = (v22 - 1) & (((unsigned int)v4 >> 4) ^ ((unsigned int)v4 >> 9));
      v25 = (_QWORD *)(v23 + 16LL * v24);
      v26 = *v25;
      if ( *v25 != v4 )
      {
        v48 = 1;
        v49 = 0;
        while ( v26 != -8 )
        {
          if ( v26 == -16 && !v49 )
            v49 = v25;
          v24 = (v22 - 1) & (v48 + v24);
          v25 = (_QWORD *)(v23 + 16LL * v24);
          v26 = *v25;
          if ( *v25 == v4 )
            goto LABEL_32;
          ++v48;
        }
        if ( v49 )
          v25 = v49;
        ++*(_QWORD *)a3;
        v46 = *(_DWORD *)(a3 + 16) + 1;
        if ( 4 * v46 < 3 * v22 )
        {
          if ( v22 - *(_DWORD *)(a3 + 20) - v46 <= v22 >> 3 )
          {
            sub_20FB870(a3, v22);
            v50 = *(_DWORD *)(a3 + 24);
            if ( !v50 )
            {
LABEL_108:
              ++*(_DWORD *)(a3 + 16);
              BUG();
            }
            v51 = v50 - 1;
            v52 = *(_QWORD *)(a3 + 8);
            v53 = 1;
            v54 = v51 & (((unsigned int)v4 >> 4) ^ ((unsigned int)v4 >> 9));
            v55 = 0;
            v46 = *(_DWORD *)(a3 + 16) + 1;
            v25 = (_QWORD *)(v52 + 16LL * v54);
            v56 = *v25;
            if ( *v25 != v4 )
            {
              while ( v56 != -8 )
              {
                if ( !v55 && v56 == -16 )
                  v55 = v25;
                v54 = v51 & (v53 + v54);
                v25 = (_QWORD *)(v52 + 16LL * v54);
                v56 = *v25;
                if ( *v25 == v4 )
                  goto LABEL_68;
                ++v53;
              }
              if ( v55 )
                v25 = v55;
            }
          }
          goto LABEL_68;
        }
LABEL_66:
        sub_20FB870(a3, 2 * v22);
        v42 = *(_DWORD *)(a3 + 24);
        if ( !v42 )
          goto LABEL_108;
        v43 = v42 - 1;
        v44 = *(_QWORD *)(a3 + 8);
        v45 = v43 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v46 = *(_DWORD *)(a3 + 16) + 1;
        v25 = (_QWORD *)(v44 + 16LL * v45);
        v47 = *v25;
        if ( *v25 != v4 )
        {
          v57 = 1;
          v58 = 0;
          while ( v47 != -8 )
          {
            if ( !v58 && v47 == -16 )
              v58 = v25;
            v45 = v43 & (v57 + v45);
            v25 = (_QWORD *)(v44 + 16LL * v45);
            v47 = *v25;
            if ( *v25 == v4 )
              goto LABEL_68;
            ++v57;
          }
          if ( v58 )
            v25 = v58;
        }
LABEL_68:
        *(_DWORD *)(a3 + 16) = v46;
        if ( *v25 != -8 )
          --*(_DWORD *)(a3 + 20);
        *v25 = v4;
        v25[1] = 0;
      }
LABEL_32:
      v27 = *(unsigned int *)(v6 + 8);
      v28 = 0;
      if ( (_DWORD)v27 == 2 )
        v28 = *(_QWORD *)(v6 - 8);
      v25[1] = sub_20FB390(a1, *(unsigned __int8 **)(v6 - 8 * v27), v28);
LABEL_12:
      result = *(_QWORD *)(v62 + 8);
      v62 = result;
      if ( v59 == result )
        return result;
    }
    ++*(_QWORD *)a3;
    goto LABEL_66;
  }
  return result;
}
