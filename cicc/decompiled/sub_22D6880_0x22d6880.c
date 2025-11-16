// Function: sub_22D6880
// Address: 0x22d6880
//
void __fastcall sub_22D6880(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rcx
  unsigned int v11; // edx
  _QWORD *v12; // r13
  __int64 v13; // rsi
  __int64 v14; // rax
  unsigned int *v15; // rax
  unsigned int *v16; // r14
  unsigned int *v17; // r13
  __int64 v18; // rcx
  unsigned int v19; // r15d
  _QWORD *v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rdi
  __int64 v23; // rsi
  __int64 v24; // rdx
  _QWORD *v25; // rdx
  int *v26; // r14
  int *v27; // r15
  unsigned int v28; // esi
  int v29; // r13d
  __int64 v30; // r9
  int v31; // ecx
  int v32; // r12d
  unsigned int v33; // r8d
  int *v34; // rdx
  int *v35; // rax
  int v36; // edi
  _QWORD *v37; // rax
  _QWORD *i; // r8
  _BYTE *v39; // rdx
  int v40; // edi
  __int64 v41; // r9
  int v42; // edi
  unsigned int v43; // esi
  _QWORD *v44; // rcx
  _BYTE *v45; // r11
  int v46; // edi
  int v47; // edx
  int v48; // eax
  __int64 v49; // rdi
  int v50; // ecx
  unsigned int v51; // edx
  int *v52; // r12
  int v53; // esi
  unsigned __int64 v54; // rdi
  int v55; // eax
  __int64 v56; // rsi
  int v57; // ecx
  unsigned int v58; // edx
  int *v59; // r12
  int v60; // edi
  unsigned __int64 v61; // rdi
  unsigned int v62; // eax
  __int64 v63; // rdx
  unsigned int v64; // eax
  unsigned int v65; // esi
  __int64 v66; // rdi
  __int64 v67; // r8
  int v68; // r8d
  int v69; // ecx
  int v70; // r10d
  int v71; // eax
  int v72; // eax
  int v73; // ecx
  int v74; // ecx
  __int64 v75; // r10
  __int64 v76; // rsi
  int v77; // edi
  int v78; // r9d
  int *v79; // r8
  int v80; // esi
  int v81; // esi
  __int64 v82; // r10
  int v83; // r9d
  __int64 v84; // rcx
  int v85; // edi
  __int64 v86; // rdx
  bool v87; // al
  bool v88; // zf
  int v89; // r8d
  __int64 v90; // [rsp+8h] [rbp-A8h]
  unsigned int v91; // [rsp+8h] [rbp-A8h]
  __int64 v92; // [rsp+10h] [rbp-A0h]
  _QWORD v93[2]; // [rsp+28h] [rbp-88h] BYREF
  __int64 v94; // [rsp+38h] [rbp-78h]
  __int64 v95; // [rsp+40h] [rbp-70h]
  int *v96; // [rsp+50h] [rbp-60h] BYREF
  __int64 v97; // [rsp+58h] [rbp-58h]
  _BYTE v98[80]; // [rsp+60h] [rbp-50h] BYREF

  v6 = a2;
  v96 = (int *)v98;
  v8 = *(_DWORD *)(a1 + 88);
  v97 = 0x800000000LL;
  v92 = a1 + 72;
  if ( v8 )
  {
    v15 = *(unsigned int **)(a1 + 80);
    v16 = &v15[22 * *(unsigned int *)(a1 + 96)];
    if ( v15 != v16 )
    {
      while ( 1 )
      {
        v17 = v15;
        if ( *v15 <= 0xFFFFFFFD )
          break;
        v15 += 22;
        if ( v16 == v15 )
          goto LABEL_2;
      }
      if ( v16 != v15 )
      {
        v18 = 0;
        v19 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
        if ( v15[6] )
          goto LABEL_70;
LABEL_21:
        v20 = (_QWORD *)*((_QWORD *)v17 + 5);
        v21 = 8LL * v17[12];
        v22 = &v20[(unsigned __int64)v21 / 8];
        v23 = v21 >> 3;
        v24 = v21 >> 5;
        if ( !v24 )
          goto LABEL_84;
        v25 = &v20[4 * v24];
        do
        {
          if ( *v20 == v6 )
            goto LABEL_28;
          if ( v20[1] == v6 )
          {
            ++v20;
            goto LABEL_28;
          }
          if ( v20[2] == v6 )
          {
            v20 += 2;
            goto LABEL_28;
          }
          if ( v20[3] == v6 )
          {
            v20 += 3;
            goto LABEL_28;
          }
          v20 += 4;
        }
        while ( v25 != v20 );
        v23 = v22 - v20;
LABEL_84:
        if ( v23 == 2 )
          goto LABEL_112;
        if ( v23 != 3 )
        {
          if ( v23 == 1 )
            goto LABEL_87;
          goto LABEL_29;
        }
        if ( *v20 == v6 )
          goto LABEL_28;
        ++v20;
LABEL_112:
        if ( *v20 == v6 )
          goto LABEL_28;
        ++v20;
LABEL_87:
        if ( *v20 == v6 )
        {
LABEL_28:
          if ( v22 != v20 )
            goto LABEL_72;
        }
LABEL_29:
        while ( 1 )
        {
          v17 += 22;
          if ( v17 == v16 )
            break;
          while ( *v17 > 0xFFFFFFFD )
          {
            v17 += 22;
            if ( v16 == v17 )
              goto LABEL_32;
          }
          if ( v16 == v17 )
            break;
          if ( !v17[6] )
            goto LABEL_21;
LABEL_70:
          v62 = v17[8];
          v63 = *((_QWORD *)v17 + 2);
          if ( v62 )
          {
            v64 = v62 - 1;
            v65 = v64 & v19;
            v66 = *(_QWORD *)(v63 + 8LL * (v64 & v19));
            if ( v6 == v66 )
            {
LABEL_72:
              v67 = *v17;
              if ( v18 + 1 > (unsigned __int64)HIDWORD(v97) )
              {
                v91 = *v17;
                sub_C8D5F0((__int64)&v96, v98, v18 + 1, 4u, v67, a6);
                v18 = (unsigned int)v97;
                LODWORD(v67) = v91;
              }
              v96[v18] = v67;
              v18 = (unsigned int)(v97 + 1);
              LODWORD(v97) = v97 + 1;
            }
            else
            {
              v68 = 1;
              while ( v66 != -4096 )
              {
                a6 = (unsigned int)(v68 + 1);
                v65 = v64 & (v68 + v65);
                v66 = *(_QWORD *)(v63 + 8LL * v65);
                if ( v66 == v6 )
                  goto LABEL_72;
                ++v68;
              }
            }
          }
        }
LABEL_32:
        v26 = v96;
        if ( v96 == &v96[(unsigned int)v18] )
          goto LABEL_2;
        v90 = v6;
        v27 = &v96[(unsigned int)v18];
        while ( 1 )
        {
          v28 = *(_DWORD *)(a1 + 96);
          v29 = *v26;
          if ( v28 )
          {
            v30 = *(_QWORD *)(a1 + 80);
            v31 = 1;
            v32 = 37 * v29;
            v33 = (v28 - 1) & (37 * v29);
            v34 = (int *)(v30 + 88LL * v33);
            v35 = 0;
            v36 = *v34;
            if ( v29 == *v34 )
            {
LABEL_36:
              v37 = (_QWORD *)*((_QWORD *)v34 + 5);
              for ( i = &v37[v34[12]]; i != v37; ++v37 )
              {
                v39 = (_BYTE *)*v37;
                if ( *(_BYTE *)*v37 == 84 )
                {
                  v40 = *(_DWORD *)(a1 + 32);
                  v41 = *(_QWORD *)(a1 + 16);
                  if ( v40 )
                  {
                    v42 = v40 - 1;
                    v43 = v42 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
                    v44 = (_QWORD *)(v41 + 16LL * v43);
                    v45 = (_BYTE *)*v44;
                    if ( v39 == (_BYTE *)*v44 )
                    {
LABEL_42:
                      *v44 = -8192;
                      --*(_DWORD *)(a1 + 24);
                      ++*(_DWORD *)(a1 + 28);
                    }
                    else
                    {
                      v69 = 1;
                      while ( v45 != (_BYTE *)-4096LL )
                      {
                        v70 = v69 + 1;
                        v43 = v42 & (v69 + v43);
                        v44 = (_QWORD *)(v41 + 16LL * v43);
                        v45 = (_BYTE *)*v44;
                        if ( v39 == (_BYTE *)*v44 )
                          goto LABEL_42;
                        v69 = v70;
                      }
                    }
                  }
                }
              }
              goto LABEL_56;
            }
            while ( v36 != -1 )
            {
              if ( v36 == -2 && !v35 )
                v35 = v34;
              v33 = (v28 - 1) & (v33 + v31);
              v34 = (int *)(v30 + 88LL * v33);
              v36 = *v34;
              if ( v29 == *v34 )
                goto LABEL_36;
              ++v31;
            }
            v46 = *(_DWORD *)(a1 + 88);
            if ( !v35 )
              v35 = v34;
            ++*(_QWORD *)(a1 + 72);
            v47 = v46 + 1;
            if ( 4 * (v46 + 1) < 3 * v28 )
            {
              if ( v28 - *(_DWORD *)(a1 + 92) - v47 > v28 >> 3 )
                goto LABEL_53;
              sub_22D6290(v92, v28);
              v80 = *(_DWORD *)(a1 + 96);
              if ( !v80 )
              {
LABEL_131:
                ++*(_DWORD *)(a1 + 88);
                BUG();
              }
              v81 = v80 - 1;
              v82 = *(_QWORD *)(a1 + 80);
              v83 = 1;
              LODWORD(v84) = v81 & v32;
              v79 = 0;
              v35 = (int *)(v82 + 88LL * (v81 & (unsigned int)v32));
              v85 = *v35;
              v47 = *(_DWORD *)(a1 + 88) + 1;
              if ( v29 == *v35 )
                goto LABEL_53;
              while ( v85 != -1 )
              {
                if ( !v79 && v85 == -2 )
                  v79 = v35;
                v84 = v81 & (unsigned int)(v84 + v83);
                v35 = (int *)(v82 + 88 * v84);
                v85 = *v35;
                if ( v29 == *v35 )
                  goto LABEL_53;
                ++v83;
              }
              goto LABEL_117;
            }
          }
          else
          {
            ++*(_QWORD *)(a1 + 72);
          }
          sub_22D6290(v92, 2 * v28);
          v73 = *(_DWORD *)(a1 + 96);
          if ( !v73 )
            goto LABEL_131;
          v74 = v73 - 1;
          v75 = *(_QWORD *)(a1 + 80);
          LODWORD(v76) = v74 & (37 * v29);
          v35 = (int *)(v75 + 88LL * (unsigned int)v76);
          v77 = *v35;
          v47 = *(_DWORD *)(a1 + 88) + 1;
          if ( v29 == *v35 )
            goto LABEL_53;
          v78 = 1;
          v79 = 0;
          while ( v77 != -1 )
          {
            if ( !v79 && v77 == -2 )
              v79 = v35;
            v76 = v74 & (unsigned int)(v76 + v78);
            v35 = (int *)(v75 + 88 * v76);
            v77 = *v35;
            if ( v29 == *v35 )
              goto LABEL_53;
            ++v78;
          }
LABEL_117:
          if ( v79 )
            v35 = v79;
LABEL_53:
          *(_DWORD *)(a1 + 88) = v47;
          if ( *v35 != -1 )
            --*(_DWORD *)(a1 + 92);
          *v35 = v29;
          *((_QWORD *)v35 + 5) = v35 + 14;
          *((_QWORD *)v35 + 6) = 0x400000000LL;
          *(_OWORD *)(v35 + 2) = 0;
          *(_OWORD *)(v35 + 6) = 0;
          *(_OWORD *)(v35 + 14) = 0;
          *(_OWORD *)(v35 + 18) = 0;
LABEL_56:
          v48 = *(_DWORD *)(a1 + 64);
          v49 = *(_QWORD *)(a1 + 48);
          if ( v48 )
          {
            v50 = v48 - 1;
            v51 = (v48 - 1) & (37 * v29);
            v52 = (int *)(v49 + 88LL * v51);
            v53 = *v52;
            if ( v29 == *v52 )
            {
LABEL_58:
              v54 = *((_QWORD *)v52 + 5);
              if ( (int *)v54 != v52 + 14 )
                _libc_free(v54);
              sub_C7D6A0(*((_QWORD *)v52 + 2), 8LL * (unsigned int)v52[8], 8);
              *v52 = -2;
              --*(_DWORD *)(a1 + 56);
              ++*(_DWORD *)(a1 + 60);
            }
            else
            {
              v72 = 1;
              while ( v53 != -1 )
              {
                v51 = v50 & (v72 + v51);
                v52 = (int *)(v49 + 88LL * v51);
                v53 = *v52;
                if ( v29 == *v52 )
                  goto LABEL_58;
                ++v72;
              }
            }
          }
          v55 = *(_DWORD *)(a1 + 96);
          v56 = *(_QWORD *)(a1 + 80);
          if ( v55 )
          {
            v57 = v55 - 1;
            v58 = (v55 - 1) & (37 * v29);
            v59 = (int *)(v56 + 88LL * v58);
            v60 = *v59;
            if ( v29 == *v59 )
            {
LABEL_63:
              v61 = *((_QWORD *)v59 + 5);
              if ( (int *)v61 != v59 + 14 )
                _libc_free(v61);
              sub_C7D6A0(*((_QWORD *)v59 + 2), 8LL * (unsigned int)v59[8], 8);
              *v59 = -2;
              --*(_DWORD *)(a1 + 88);
              ++*(_DWORD *)(a1 + 92);
            }
            else
            {
              v71 = 1;
              while ( v60 != -1 )
              {
                v58 = v57 & (v71 + v58);
                v59 = (int *)(v56 + 88LL * v58);
                v60 = *v59;
                if ( v29 == *v59 )
                  goto LABEL_63;
                ++v71;
              }
            }
          }
          if ( v27 == ++v26 )
          {
            v6 = v90;
            break;
          }
        }
      }
    }
  }
LABEL_2:
  v9 = *(unsigned int *)(a1 + 128);
  if ( (_DWORD)v9 )
  {
    v10 = *(_QWORD *)(a1 + 112);
    v11 = (v9 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v12 = (_QWORD *)(v10 + 40LL * v11);
    v13 = v12[3];
    if ( v13 == v6 )
    {
LABEL_4:
      if ( v12 != (_QWORD *)(v10 + 40 * v9) )
      {
        v94 = -8192;
        v95 = 0;
        v93[1] = 0;
        v14 = v12[3];
        v93[0] = 2;
        if ( v14 == -8192 )
        {
          v12[4] = 0;
        }
        else if ( v14 == -4096 || !v14 )
        {
          v12[3] = -8192;
          v86 = v94;
          v87 = v94 != 0;
          v88 = v94 == -4096;
          v12[4] = v95;
          if ( v86 != -8192 && !v88 && v87 )
            sub_BD60C0(v93);
        }
        else
        {
          sub_BD60C0(v12 + 1);
          v12[3] = v94;
          v12[4] = v95;
        }
        --*(_DWORD *)(a1 + 120);
        ++*(_DWORD *)(a1 + 124);
      }
    }
    else
    {
      v89 = 1;
      while ( v13 != -4096 )
      {
        v11 = (v9 - 1) & (v89 + v11);
        v12 = (_QWORD *)(v10 + 40LL * v11);
        v13 = v12[3];
        if ( v13 == v6 )
          goto LABEL_4;
        ++v89;
      }
    }
  }
  if ( v96 != (int *)v98 )
    _libc_free((unsigned __int64)v96);
}
