// Function: sub_2B66F90
// Address: 0x2b66f90
//
__int64 __fastcall sub_2B66F90(__int64 *a1, unsigned __int8 **a2, unsigned __int64 a3)
{
  unsigned __int8 **v5; // r12
  unsigned __int8 **v6; // rbx
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rdx
  unsigned int v11; // eax
  unsigned __int8 *v12; // rsi
  int v13; // eax
  __int64 v14; // r13
  _DWORD *v15; // rbx
  __int64 v16; // rdi
  int v17; // r8d
  __int64 v18; // rdi
  _QWORD *v19; // r9
  int v20; // r8d
  __int64 v21; // r10
  _DWORD *v22; // rax
  unsigned __int8 *v23; // rsi
  int v24; // edx
  int v25; // eax
  __int64 v26; // r15
  unsigned int v27; // r14d
  unsigned int v28; // esi
  __int64 v29; // r8
  unsigned int v30; // edi
  unsigned int *v31; // rax
  int v32; // r11d
  unsigned int *v33; // r10
  int v34; // ecx
  __int64 v35; // rdx
  __int64 v36; // r8
  unsigned int v37; // r9d
  __int64 v38; // rdx
  __int64 v39; // rcx
  unsigned int v40; // r14d
  __int64 v41; // rdx
  __int64 v42; // rdi
  int v43; // esi
  unsigned int v44; // eax
  __int64 v45; // r8
  unsigned int v46; // eax
  __int64 v47; // rsi
  int v48; // edi
  int v50; // esi
  _QWORD *v51; // rax
  unsigned __int8 *v52; // rdx
  _DWORD *v53; // r12
  __int64 v54; // r11
  __int64 v55; // rsi
  int v56; // r8d
  int v57; // r8d
  __int64 v58; // r9
  unsigned int v59; // edx
  unsigned int v60; // r11d
  int v61; // edi
  unsigned int *v62; // rsi
  int v63; // r9d
  int v64; // r9d
  __int64 v65; // r10
  int v66; // edi
  __int64 v67; // rdx
  unsigned int v68; // r8d
  int v69; // [rsp+Ch] [rbp-64h]
  int v70; // [rsp+Ch] [rbp-64h]
  unsigned __int8 *v71; // [rsp+18h] [rbp-58h] BYREF
  __int64 v72; // [rsp+20h] [rbp-50h] BYREF
  _DWORD *v73; // [rsp+28h] [rbp-48h]
  __int64 v74; // [rsp+30h] [rbp-40h]
  unsigned int v75; // [rsp+38h] [rbp-38h]

  v5 = &a2[a3];
  v6 = a2;
  if ( v5 != sub_2B0BF30(a2, (__int64)v5, (unsigned __int8 (__fastcall *)(_QWORD))sub_2B0D8B0)
    && (sub_2B08550(a2, a3)
     || !(unsigned __int8)sub_2B17600(a2, a3)
     || !(unsigned __int8)sub_2B0D770(a2, a3, v35, v7, v36, v37)
     || !sub_2B5F980((__int64 *)a2, a3, *(__int64 **)(*a1 + 3304))
     || !v38) )
  {
    v72 = 0;
    v73 = 0;
    v74 = 0;
    v75 = 0;
    if ( v5 == a2 )
    {
      v53 = 0;
      v54 = 0;
      v55 = 0;
      goto LABEL_87;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v12 = *v6;
        v71 = v12;
        v13 = *v12;
        if ( (unsigned __int8)v13 <= 0x15u )
          goto LABEL_11;
        if ( (_BYTE)v13 != 90 )
          break;
LABEL_9:
        if ( v5 == ++v6 )
          goto LABEL_13;
      }
      v8 = *a1;
      if ( (*(_BYTE *)(*a1 + 88) & 1) != 0 )
      {
        v9 = v8 + 96;
        v10 = 3;
      }
      else
      {
        v10 = *(unsigned int *)(v8 + 104);
        v9 = *(_QWORD *)(v8 + 96);
        if ( !(_DWORD)v10 )
          goto LABEL_24;
        v10 = (unsigned int)(v10 - 1);
      }
      v11 = v10 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v7 = *(_QWORD *)(v9 + 72LL * v11);
      if ( v12 == (unsigned __int8 *)v7 )
        goto LABEL_9;
      v17 = 1;
      while ( v7 != -4096 )
      {
        v11 = v10 & (v17 + v11);
        v7 = *(_QWORD *)(v9 + 72LL * v11);
        if ( v12 == (unsigned __int8 *)v7 )
          goto LABEL_9;
        ++v17;
      }
LABEL_24:
      v18 = *(_QWORD *)a1[1];
      if ( !v18 || !(unsigned __int8)sub_D48480(v18, (__int64)v12, v10, v7) )
      {
        if ( v75 )
        {
          v7 = (__int64)v71;
          v19 = 0;
          v20 = 1;
          LODWORD(v21) = (v75 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
          v22 = &v73[4 * (unsigned int)v21];
          v23 = *(unsigned __int8 **)v22;
          if ( v71 == *(unsigned __int8 **)v22 )
          {
LABEL_28:
            v24 = v22[2];
            if ( v24 == 1 )
            {
              ++*(_DWORD *)a1[3];
              v7 = (__int64)v71;
              v24 = v22[2];
            }
            v22[2] = v24 + 1;
            v25 = *(unsigned __int8 *)v7;
            if ( (unsigned __int8)v25 <= 0x1Cu )
              goto LABEL_9;
LABEL_31:
            v26 = a1[4];
            v27 = v25 - 29;
            v28 = *(_DWORD *)(v26 + 24);
            if ( !v28 )
            {
              ++*(_QWORD *)v26;
              goto LABEL_93;
            }
            v29 = *(_QWORD *)(v26 + 8);
            v30 = (v28 - 1) & (37 * v27);
            v31 = (unsigned int *)(v29 + 4LL * v30);
            v7 = *v31;
            if ( (_DWORD)v7 != v27 )
            {
              v32 = 1;
              v33 = 0;
              while ( (_DWORD)v7 != -1 )
              {
                if ( (_DWORD)v7 == -2 && !v33 )
                  v33 = v31;
                v30 = (v28 - 1) & (v32 + v30);
                v31 = (unsigned int *)(v29 + 4LL * v30);
                v7 = *v31;
                if ( v27 == (_DWORD)v7 )
                  goto LABEL_9;
                ++v32;
              }
              v34 = *(_DWORD *)(v26 + 16);
              if ( v33 )
                v31 = v33;
              ++*(_QWORD *)v26;
              v7 = (unsigned int)(v34 + 1);
              if ( 4 * (int)v7 < 3 * v28 )
              {
                if ( v28 - *(_DWORD *)(v26 + 20) - (unsigned int)v7 <= v28 >> 3 )
                {
                  v70 = 37 * v27;
                  sub_A08C50(v26, v28);
                  v63 = *(_DWORD *)(v26 + 24);
                  if ( !v63 )
                  {
LABEL_118:
                    ++*(_DWORD *)(v26 + 16);
                    BUG();
                  }
                  v64 = v63 - 1;
                  v65 = *(_QWORD *)(v26 + 8);
                  v66 = 1;
                  v62 = 0;
                  LODWORD(v67) = v64 & v70;
                  v7 = (unsigned int)(*(_DWORD *)(v26 + 16) + 1);
                  v31 = (unsigned int *)(v65 + 4LL * (v64 & (unsigned int)v70));
                  v68 = *v31;
                  if ( v27 != *v31 )
                  {
                    while ( v68 != -1 )
                    {
                      if ( !v62 && v68 == -2 )
                        v62 = v31;
                      v67 = v64 & (unsigned int)(v67 + v66);
                      v31 = (unsigned int *)(v65 + 4 * v67);
                      v68 = *v31;
                      if ( v27 == *v31 )
                        goto LABEL_39;
                      ++v66;
                    }
                    goto LABEL_97;
                  }
                }
                goto LABEL_39;
              }
LABEL_93:
              sub_A08C50(v26, 2 * v28);
              v56 = *(_DWORD *)(v26 + 24);
              if ( !v56 )
                goto LABEL_118;
              v57 = v56 - 1;
              v58 = *(_QWORD *)(v26 + 8);
              v59 = v57 & (37 * v27);
              v7 = (unsigned int)(*(_DWORD *)(v26 + 16) + 1);
              v31 = (unsigned int *)(v58 + 4LL * v59);
              v60 = *v31;
              if ( v27 != *v31 )
              {
                v61 = 1;
                v62 = 0;
                while ( v60 != -1 )
                {
                  if ( v60 == -2 && !v62 )
                    v62 = v31;
                  v59 = v57 & (v61 + v59);
                  v31 = (unsigned int *)(v58 + 4LL * v59);
                  v60 = *v31;
                  if ( v27 == *v31 )
                    goto LABEL_39;
                  ++v61;
                }
LABEL_97:
                if ( v62 )
                  v31 = v62;
              }
LABEL_39:
              *(_DWORD *)(v26 + 16) = v7;
              if ( *v31 != -1 )
                --*(_DWORD *)(v26 + 20);
              *v31 = v27;
              goto LABEL_9;
            }
            goto LABEL_9;
          }
          while ( v23 != (unsigned __int8 *)-4096LL )
          {
            if ( v23 == (unsigned __int8 *)-8192LL && !v19 )
              v19 = v22;
            v21 = (v75 - 1) & ((_DWORD)v21 + v20);
            v22 = &v73[4 * v21];
            v23 = *(unsigned __int8 **)v22;
            if ( v71 == *(unsigned __int8 **)v22 )
              goto LABEL_28;
            ++v20;
          }
          if ( !v19 )
            v19 = v22;
        }
        else
        {
          v19 = 0;
        }
        v51 = sub_10E8350((__int64)&v72, &v71, v19);
        v52 = v71;
        *((_DWORD *)v51 + 2) = 1;
        *v51 = v52;
        v25 = *v52;
        if ( (unsigned __int8)v25 <= 0x1Cu )
        {
          ++*(_DWORD *)a1[5];
          goto LABEL_9;
        }
        goto LABEL_31;
      }
      v13 = *v71;
LABEL_11:
      if ( (unsigned int)(v13 - 12) > 1 )
        goto LABEL_9;
      ++v6;
      ++*(_DWORD *)a1[2];
      if ( v5 == v6 )
      {
LABEL_13:
        v54 = (__int64)v73;
        v14 = *a1;
        v55 = 4LL * v75;
        v53 = &v73[v55];
        if ( (_DWORD)v74 && v53 != v73 )
        {
          v15 = v73;
          while ( 1 )
          {
            v16 = *(_QWORD *)v15;
            if ( *(_QWORD *)v15 != -8192 && v16 != -4096 )
              break;
            v15 += 4;
            if ( v53 == v15 )
              goto LABEL_87;
          }
          if ( v53 == v15 )
            goto LABEL_68;
LABEL_49:
          if ( !(unsigned __int8)sub_BD3660(v16, v15[2] + 1) )
            goto LABEL_50;
          v54 = (__int64)v73;
          v39 = *(_QWORD *)(*(_QWORD *)v15 + 16LL);
          if ( !v39 )
          {
LABEL_67:
            v55 = 4LL * v75;
LABEL_68:
            LOBYTE(v53) = v53 == v15;
            sub_C7D6A0(v54, v55 * 4, 8);
            return (unsigned int)v53;
          }
          v40 = v75 - 1;
          while ( 2 )
          {
            v41 = *(_QWORD *)(v39 + 24);
            if ( (*(_BYTE *)(v14 + 88) & 1) != 0 )
            {
              v42 = v14 + 96;
              v43 = 3;
LABEL_59:
              v44 = v43 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
              v45 = *(_QWORD *)(v42 + 72LL * v44);
              if ( v41 == v45 )
                goto LABEL_50;
              v69 = 1;
              while ( v45 != -4096 )
              {
                v44 = v43 & (v69 + v44);
                ++v69;
                v45 = *(_QWORD *)(v42 + 72LL * v44);
                if ( v41 == v45 )
                  goto LABEL_50;
              }
            }
            else
            {
              v50 = *(_DWORD *)(v14 + 104);
              v42 = *(_QWORD *)(v14 + 96);
              if ( v50 )
              {
                v43 = v50 - 1;
                goto LABEL_59;
              }
            }
            if ( v75 )
            {
              v46 = v40 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
              v47 = *(_QWORD *)&v73[4 * v46];
              if ( v41 == v47 )
                goto LABEL_50;
              v48 = 1;
              while ( v47 != -4096 )
              {
                v46 = v40 & (v48 + v46);
                v47 = *(_QWORD *)&v73[4 * v46];
                if ( v41 == v47 )
                {
                  do
                  {
LABEL_50:
                    v15 += 4;
                    if ( v53 == v15 )
                    {
                      v54 = (__int64)v73;
                      v15 = v53;
                      v55 = 4LL * v75;
                      goto LABEL_68;
                    }
                    v16 = *(_QWORD *)v15;
                  }
                  while ( *(_QWORD *)v15 == -8192 || v16 == -4096 );
                  if ( v53 != v15 )
                    goto LABEL_49;
                  v54 = (__int64)v73;
                  v55 = 4LL * v75;
                  goto LABEL_68;
                }
                ++v48;
              }
            }
            v39 = *(_QWORD *)(v39 + 8);
            if ( !v39 )
              goto LABEL_67;
            continue;
          }
        }
LABEL_87:
        v15 = v53;
        goto LABEL_68;
      }
    }
  }
  LODWORD(v53) = 0;
  return (unsigned int)v53;
}
