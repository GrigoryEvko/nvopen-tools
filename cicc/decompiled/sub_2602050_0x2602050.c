// Function: sub_2602050
// Address: 0x2602050
//
__int64 __fastcall sub_2602050(__int64 ***a1, __int64 a2)
{
  int v2; // r15d
  __int64 **v3; // rax
  __int64 **v4; // rdx
  unsigned int v6; // r10d
  __int64 v7; // r11
  _BYTE **v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r14
  _BYTE **v11; // r11
  __int64 v12; // rax
  _BYTE *v13; // r12
  __int64 v14; // rsi
  unsigned int v15; // ecx
  __int64 v16; // rdx
  _BYTE *v17; // rdi
  unsigned int v18; // esi
  __int64 v19; // rax
  unsigned int v20; // edx
  int *v21; // rcx
  int v22; // edi
  __int64 v23; // rdx
  int v24; // edi
  unsigned int v25; // edi
  _DWORD *v26; // rcx
  int v27; // edx
  _DWORD *v28; // r9
  int v29; // eax
  int v30; // eax
  int v32; // edx
  int v33; // r9d
  int v34; // ecx
  int v35; // r9d
  __int64 v36; // rcx
  int v37; // eax
  int v38; // eax
  int v39; // ecx
  __int64 v40; // rdi
  __int64 v41; // rdx
  int v42; // esi
  int v43; // r10d
  _DWORD *v44; // r8
  int v45; // eax
  int v46; // edx
  __int64 v47; // rsi
  _DWORD *v48; // rdi
  unsigned int v49; // r12d
  int v50; // r8d
  int v51; // ecx
  unsigned int v52; // esi
  int v53; // edi
  int v54; // r10d
  __int64 v55; // r9
  int v56; // r10d
  unsigned int v57; // r8d
  int v58; // esi
  int v59; // r10d
  unsigned int v60; // r8d
  __int64 **v61; // [rsp+0h] [rbp-90h]
  int v62; // [rsp+10h] [rbp-80h]
  __int64 **v63; // [rsp+18h] [rbp-78h]
  __int64 v64; // [rsp+20h] [rbp-70h]
  unsigned int v65; // [rsp+28h] [rbp-68h]
  int v66; // [rsp+28h] [rbp-68h]
  _BYTE **v67; // [rsp+28h] [rbp-68h]
  __int64 v68; // [rsp+30h] [rbp-60h]
  __int64 v69; // [rsp+38h] [rbp-58h]
  _BYTE **v70; // [rsp+38h] [rbp-58h]
  _BYTE **v71; // [rsp+38h] [rbp-58h]
  _BYTE **v72; // [rsp+38h] [rbp-58h]
  __int64 v73; // [rsp+40h] [rbp-50h] BYREF
  __int64 v74; // [rsp+48h] [rbp-48h]
  __int64 v75; // [rsp+50h] [rbp-40h]
  unsigned int v76; // [rsp+58h] [rbp-38h]

  v3 = a1[1];
  v4 = *a1;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v61 = v3;
  if ( v4 != v3 )
  {
    v63 = v4;
    v69 = 0;
    v6 = 0;
    while ( 1 )
    {
      v7 = **v63;
      v68 = *(_QWORD *)(v7 + 8);
      v64 = *(_QWORD *)(*(_QWORD *)(v7 + 16) + 8LL);
      if ( v68 == v64 )
        goto LABEL_31;
      do
      {
        v8 = *(_BYTE ***)(v68 + 24);
        v9 = *(unsigned int *)(v68 + 32);
        if ( v8 == &v8[v9] )
          goto LABEL_30;
        v10 = v7;
        v11 = &v8[v9];
        do
        {
          while ( 1 )
          {
            v12 = *(unsigned int *)(v10 + 48);
            v13 = *v8;
            v14 = *(_QWORD *)(v10 + 32);
            if ( (_DWORD)v12 )
            {
              v15 = (v12 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
              v16 = v14 + 16LL * v15;
              v17 = *(_BYTE **)v16;
              if ( v13 == *(_BYTE **)v16 )
              {
LABEL_9:
                if ( v16 != v14 + 16 * v12 )
                  v2 = *(_DWORD *)(v16 + 8);
              }
              else
              {
                v32 = 1;
                while ( v17 != (_BYTE *)-4096LL )
                {
                  v33 = v32 + 1;
                  v15 = (v12 - 1) & (v32 + v15);
                  v16 = v14 + 16LL * v15;
                  v17 = *(_BYTE **)v16;
                  if ( v13 == *(_BYTE **)v16 )
                    goto LABEL_9;
                  v32 = v33;
                }
              }
            }
            v18 = *(_DWORD *)(a2 + 24);
            v19 = *(_QWORD *)(a2 + 8);
            if ( v18 )
            {
              v20 = (v18 - 1) & (37 * v2);
              v21 = (int *)(v19 + 4LL * v20);
              v22 = *v21;
              if ( v2 == *v21 )
              {
LABEL_13:
                if ( v21 != (int *)(v19 + 4LL * v18) )
                  goto LABEL_6;
              }
              else
              {
                v34 = 1;
                while ( v22 != -1 )
                {
                  v35 = v34 + 1;
                  v20 = (v18 - 1) & (v34 + v20);
                  v21 = (int *)(v19 + 4LL * v20);
                  v22 = *v21;
                  if ( v2 == *v21 )
                    goto LABEL_13;
                  v34 = v35;
                }
              }
            }
            if ( *v13 > 0x15u )
              goto LABEL_111;
            if ( !v6 )
            {
              ++v73;
              goto LABEL_65;
            }
            v65 = (v6 - 1) & (37 * v2);
            v23 = v69 + 16LL * v65;
            v24 = *(_DWORD *)v23;
            if ( v2 != *(_DWORD *)v23 )
            {
              v62 = 1;
              v36 = 0;
              while ( v24 != -1 )
              {
                if ( v24 == -2 && !v36 )
                  v36 = v23;
                v65 = (v6 - 1) & (v62 + v65);
                v23 = v69 + 16LL * v65;
                v24 = *(_DWORD *)v23;
                if ( v2 == *(_DWORD *)v23 )
                  goto LABEL_17;
                ++v62;
              }
              if ( !v36 )
                v36 = v23;
              ++v73;
              v37 = v75 + 1;
              if ( 4 * ((int)v75 + 1) < 3 * v6 )
              {
                if ( v6 - (v37 + HIDWORD(v75)) <= v6 >> 3 )
                {
                  v67 = v11;
                  sub_2601E70((__int64)&v73, v6);
                  if ( !v76 )
                  {
LABEL_107:
                    LODWORD(v75) = v75 + 1;
                    BUG();
                  }
                  v55 = 0;
                  v11 = v67;
                  v56 = 1;
                  v57 = (v76 - 1) & (37 * v2);
                  v37 = v75 + 1;
                  v36 = v74 + 16LL * v57;
                  v58 = *(_DWORD *)v36;
                  if ( v2 != *(_DWORD *)v36 )
                  {
                    while ( v58 != -1 )
                    {
                      if ( v58 == -2 && !v55 )
                        v55 = v36;
                      v57 = (v76 - 1) & (v56 + v57);
                      v36 = v74 + 16LL * v57;
                      v58 = *(_DWORD *)v36;
                      if ( v2 == *(_DWORD *)v36 )
                        goto LABEL_47;
                      ++v56;
                    }
                    goto LABEL_69;
                  }
                }
                goto LABEL_47;
              }
LABEL_65:
              v72 = v11;
              sub_2601E70((__int64)&v73, 2 * v6);
              if ( !v76 )
                goto LABEL_107;
              v11 = v72;
              v52 = (v76 - 1) & (37 * v2);
              v37 = v75 + 1;
              v36 = v74 + 16LL * v52;
              v53 = *(_DWORD *)v36;
              if ( v2 != *(_DWORD *)v36 )
              {
                v54 = 1;
                v55 = 0;
                while ( v53 != -1 )
                {
                  if ( v53 == -2 && !v55 )
                    v55 = v36;
                  v52 = (v76 - 1) & (v54 + v52);
                  v36 = v74 + 16LL * v52;
                  v53 = *(_DWORD *)v36;
                  if ( v2 == *(_DWORD *)v36 )
                    goto LABEL_47;
                  ++v54;
                }
LABEL_69:
                if ( v55 )
                  v36 = v55;
              }
LABEL_47:
              LODWORD(v75) = v37;
              if ( *(_DWORD *)v36 != -1 )
                --HIDWORD(v75);
              *(_DWORD *)v36 = v2;
              *(_QWORD *)(v36 + 8) = v13;
              v6 = v76;
              v69 = v74;
              goto LABEL_6;
            }
LABEL_17:
            if ( v13 != *(_BYTE **)(v23 + 8) )
            {
LABEL_111:
              if ( !v18 )
              {
                ++*(_QWORD *)a2;
LABEL_51:
                v70 = v11;
                sub_A08C50(a2, 2 * v18);
                v38 = *(_DWORD *)(a2 + 24);
                if ( !v38 )
                  goto LABEL_108;
                v39 = v38 - 1;
                v40 = *(_QWORD *)(a2 + 8);
                v11 = v70;
                LODWORD(v41) = (v38 - 1) & (37 * v2);
                v28 = (_DWORD *)(v40 + 4LL * (unsigned int)v41);
                v42 = *v28;
                v30 = *(_DWORD *)(a2 + 16) + 1;
                if ( v2 != *v28 )
                {
                  v43 = 1;
                  v44 = 0;
                  while ( v42 != -1 )
                  {
                    if ( v42 == -2 && !v44 )
                      v44 = v28;
                    v41 = v39 & (unsigned int)(v41 + v43);
                    v28 = (_DWORD *)(v40 + 4 * v41);
                    v42 = *v28;
                    if ( v2 == *v28 )
                      goto LABEL_26;
                    ++v43;
                  }
                  if ( v44 )
                    v28 = v44;
                }
                goto LABEL_26;
              }
              v25 = (v18 - 1) & (37 * v2);
              v26 = (_DWORD *)(v19 + 4LL * v25);
              v27 = *v26;
              if ( v2 != *v26 )
                break;
            }
LABEL_6:
            if ( v11 == ++v8 )
              goto LABEL_29;
          }
          v66 = 1;
          v28 = 0;
          while ( v27 != -1 )
          {
            if ( v28 || v27 != -2 )
              v26 = v28;
            v25 = (v18 - 1) & (v66 + v25);
            v27 = *(_DWORD *)(v19 + 4LL * v25);
            if ( v2 == v27 )
              goto LABEL_6;
            ++v66;
            v28 = v26;
            v26 = (_DWORD *)(v19 + 4LL * v25);
          }
          v29 = *(_DWORD *)(a2 + 16);
          if ( !v28 )
            v28 = v26;
          ++*(_QWORD *)a2;
          v30 = v29 + 1;
          if ( 4 * v30 >= 3 * v18 )
            goto LABEL_51;
          if ( v18 - (v30 + *(_DWORD *)(a2 + 20)) <= v18 >> 3 )
          {
            v71 = v11;
            sub_A08C50(a2, v18);
            v45 = *(_DWORD *)(a2 + 24);
            if ( !v45 )
            {
LABEL_108:
              ++*(_DWORD *)(a2 + 16);
              BUG();
            }
            v46 = v45 - 1;
            v47 = *(_QWORD *)(a2 + 8);
            v48 = 0;
            v11 = v71;
            v49 = (v45 - 1) & (37 * v2);
            v50 = 1;
            v28 = (_DWORD *)(v47 + 4LL * v49);
            v51 = *v28;
            v30 = *(_DWORD *)(a2 + 16) + 1;
            if ( v2 != *v28 )
            {
              while ( v51 != -1 )
              {
                if ( !v48 && v51 == -2 )
                  v48 = v28;
                v59 = v50 + 1;
                v60 = v46 & (v49 + v50);
                v28 = (_DWORD *)(v47 + 4LL * v60);
                v49 = v60;
                v51 = *v28;
                if ( v2 == *v28 )
                  goto LABEL_26;
                v50 = v59;
              }
              if ( v48 )
                v28 = v48;
            }
          }
LABEL_26:
          *(_DWORD *)(a2 + 16) = v30;
          if ( *v28 != -1 )
            --*(_DWORD *)(a2 + 20);
          *v28 = v2;
          ++v8;
          v6 = v76;
          v69 = v74;
        }
        while ( v11 != v8 );
LABEL_29:
        v7 = v10;
LABEL_30:
        v68 = *(_QWORD *)(v68 + 8);
      }
      while ( v68 != v64 );
LABEL_31:
      if ( v61 == ++v63 )
        return sub_C7D6A0(v69, 16LL * v6, 8);
    }
  }
  return sub_C7D6A0(0, 0, 8);
}
