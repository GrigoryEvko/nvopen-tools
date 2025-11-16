// Function: sub_FAEA90
// Address: 0xfaea90
//
__int64 __fastcall sub_FAEA90(__int64 *a1, __int64 *a2)
{
  unsigned int v2; // r14d
  __int64 v4; // rcx
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  int v7; // edx
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // r12
  __int64 v13; // r14
  __int64 v14; // r8
  int v15; // esi
  unsigned int v16; // ecx
  _QWORD *v17; // rdx
  __int64 v18; // r9
  __int64 v19; // r15
  __int64 v20; // r8
  int v21; // esi
  unsigned int v22; // ecx
  _QWORD *v23; // rdx
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r15
  unsigned int v28; // esi
  __int64 v29; // r8
  int v30; // r10d
  unsigned int v31; // edi
  __int64 *v32; // rdx
  __int64 *v33; // rax
  __int64 v34; // rcx
  __int64 v35; // rbx
  char v36; // al
  unsigned int v37; // esi
  int v39; // edi
  int v40; // ecx
  _QWORD *v41; // rdx
  _QWORD *v42; // rax
  unsigned int v43; // esi
  unsigned int v44; // edx
  int v45; // ecx
  unsigned int v46; // edi
  __int64 *v47; // rax
  __int64 v48; // rdx
  unsigned int v49; // edx
  int v50; // ecx
  unsigned int v51; // edi
  __int64 *v52; // rax
  __int64 v53; // rdx
  int v54; // r11d
  _QWORD *v55; // r10
  int v56; // r11d
  _QWORD *v57; // r10
  int v58; // eax
  int v59; // esi
  __int64 v60; // rdi
  __int64 v61; // rdx
  __int64 v62; // r8
  int v63; // r10d
  __int64 *v64; // r9
  int v65; // eax
  int v66; // esi
  __int64 v67; // rdi
  __int64 *v68; // r8
  __int64 v69; // rbx
  int v70; // r9d
  __int64 v71; // rdx
  __int64 v72; // [rsp+18h] [rbp-48h] BYREF
  __int64 v73; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v74[7]; // [rsp+28h] [rbp-38h] BYREF

  if ( a1 == (__int64 *)-4096LL
    || a2 == (__int64 *)-4096LL
    || (LOBYTE(v2) = a2 + 1024 == 0 || a1 + 1024 == 0, (_BYTE)v2) )
  {
    LOBYTE(v2) = a1 == a2;
    return v2;
  }
  v4 = *a2;
  v5 = *(_QWORD *)(*a1 + 48);
  v72 = *a1;
  v73 = v4;
  v6 = v5 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6 == v72 + 48 )
  {
    v8 = 0;
  }
  else
  {
    if ( !v6 )
      goto LABEL_124;
    v7 = *(unsigned __int8 *)(v6 - 24);
    v8 = v6 - 24;
    if ( (unsigned int)(v7 - 30) >= 0xB )
      v8 = 0;
  }
  v9 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v9 == v4 + 48 )
    goto LABEL_122;
  if ( !v9 )
LABEL_124:
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 > 0xA )
LABEL_122:
    BUG();
  v10 = *(_QWORD *)(v8 - 32);
  if ( v10 == *(_QWORD *)(v9 - 56) )
  {
    v12 = sub_AA5930(v10);
    if ( v11 != v12 )
    {
      v13 = v11;
      do
      {
        v27 = a1[1];
        v28 = *(_DWORD *)(v27 + 24);
        if ( v28 )
        {
          v29 = *(_QWORD *)(v27 + 8);
          v30 = 1;
          v31 = (v28 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v32 = (__int64 *)(v29 + 152LL * v31);
          v33 = 0;
          v34 = *v32;
          if ( v12 == *v32 )
          {
LABEL_29:
            v35 = (__int64)(v32 + 1);
            goto LABEL_30;
          }
          while ( v34 != -4096 )
          {
            if ( !v33 && v34 == -8192 )
              v33 = v32;
            v31 = (v28 - 1) & (v30 + v31);
            v32 = (__int64 *)(v29 + 152LL * v31);
            v34 = *v32;
            if ( *v32 == v12 )
              goto LABEL_29;
            ++v30;
          }
          v39 = *(_DWORD *)(v27 + 16);
          if ( !v33 )
            v33 = v32;
          ++*(_QWORD *)v27;
          v40 = v39 + 1;
          if ( 4 * (v39 + 1) < 3 * v28 )
          {
            if ( v28 - *(_DWORD *)(v27 + 20) - v40 <= v28 >> 3 )
            {
              sub_F9EFC0(v27, v28);
              v65 = *(_DWORD *)(v27 + 24);
              if ( !v65 )
              {
LABEL_121:
                ++*(_DWORD *)(v27 + 16);
                BUG();
              }
              v66 = v65 - 1;
              v67 = *(_QWORD *)(v27 + 8);
              v68 = 0;
              LODWORD(v69) = (v65 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
              v70 = 1;
              v40 = *(_DWORD *)(v27 + 16) + 1;
              v33 = (__int64 *)(v67 + 152LL * (unsigned int)v69);
              v71 = *v33;
              if ( *v33 != v12 )
              {
                while ( v71 != -4096 )
                {
                  if ( v71 == -8192 && !v68 )
                    v68 = v33;
                  v69 = v66 & (unsigned int)(v69 + v70);
                  v33 = (__int64 *)(v67 + 152 * v69);
                  v71 = *v33;
                  if ( *v33 == v12 )
                    goto LABEL_45;
                  ++v70;
                }
                if ( v68 )
                  v33 = v68;
              }
            }
            goto LABEL_45;
          }
        }
        else
        {
          ++*(_QWORD *)v27;
        }
        sub_F9EFC0(v27, 2 * v28);
        v58 = *(_DWORD *)(v27 + 24);
        if ( !v58 )
          goto LABEL_121;
        v59 = v58 - 1;
        v60 = *(_QWORD *)(v27 + 8);
        LODWORD(v61) = (v58 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v40 = *(_DWORD *)(v27 + 16) + 1;
        v33 = (__int64 *)(v60 + 152LL * (unsigned int)v61);
        v62 = *v33;
        if ( *v33 != v12 )
        {
          v63 = 1;
          v64 = 0;
          while ( v62 != -4096 )
          {
            if ( !v64 && v62 == -8192 )
              v64 = v33;
            v61 = v59 & (unsigned int)(v61 + v63);
            v33 = (__int64 *)(v60 + 152 * v61);
            v62 = *v33;
            if ( *v33 == v12 )
              goto LABEL_45;
            ++v63;
          }
          if ( v64 )
            v33 = v64;
        }
LABEL_45:
        *(_DWORD *)(v27 + 16) = v40;
        if ( *v33 != -4096 )
          --*(_DWORD *)(v27 + 20);
        *v33 = v12;
        v35 = (__int64)(v33 + 1);
        v41 = v33 + 3;
        v42 = v33 + 19;
        *(v42 - 18) = 0;
        *(v42 - 17) = 1;
        do
        {
          if ( v41 )
            *v41 = -4096;
          v41 += 2;
        }
        while ( v41 != v42 );
LABEL_30:
        v36 = *(_BYTE *)(v35 + 8) & 1;
        if ( v36 )
        {
          v14 = v35 + 16;
          v15 = 7;
        }
        else
        {
          v37 = *(_DWORD *)(v35 + 24);
          v14 = *(_QWORD *)(v35 + 16);
          if ( !v37 )
          {
            v74[0] = 0;
            v44 = *(_DWORD *)(v35 + 8);
            ++*(_QWORD *)v35;
            v45 = (v44 >> 1) + 1;
LABEL_55:
            v46 = 3 * v37;
            goto LABEL_56;
          }
          v15 = v37 - 1;
        }
        v16 = v15 & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
        v17 = (_QWORD *)(v14 + 16LL * v16);
        v18 = *v17;
        if ( v72 == *v17 )
        {
LABEL_16:
          v19 = v17[1];
          goto LABEL_17;
        }
        v56 = 1;
        v57 = 0;
        while ( v18 != -4096 )
        {
          if ( v18 == -8192 && !v57 )
            v57 = v17;
          v16 = v15 & (v56 + v16);
          v17 = (_QWORD *)(v14 + 16LL * v16);
          v18 = *v17;
          if ( v72 == *v17 )
            goto LABEL_16;
          ++v56;
        }
        v46 = 24;
        v37 = 8;
        if ( !v57 )
          v57 = v17;
        v74[0] = v57;
        v44 = *(_DWORD *)(v35 + 8);
        ++*(_QWORD *)v35;
        v45 = (v44 >> 1) + 1;
        if ( !v36 )
        {
          v37 = *(_DWORD *)(v35 + 24);
          goto LABEL_55;
        }
LABEL_56:
        if ( 4 * v45 >= v46 )
        {
          v37 *= 2;
LABEL_71:
          sub_FADF20(v35, v37);
          sub_F9DCC0(v35, &v72, v74);
          v44 = *(_DWORD *)(v35 + 8);
          goto LABEL_58;
        }
        if ( v37 - *(_DWORD *)(v35 + 12) - v45 <= v37 >> 3 )
          goto LABEL_71;
LABEL_58:
        *(_DWORD *)(v35 + 8) = (2 * (v44 >> 1) + 2) | v44 & 1;
        v47 = (__int64 *)v74[0];
        if ( *(_QWORD *)v74[0] != -4096 )
          --*(_DWORD *)(v35 + 12);
        v48 = v72;
        v47[1] = 0;
        v19 = 0;
        *v47 = v48;
        v36 = *(_BYTE *)(v35 + 8) & 1;
LABEL_17:
        if ( v36 )
        {
          v20 = v35 + 16;
          v21 = 7;
        }
        else
        {
          v43 = *(_DWORD *)(v35 + 24);
          v20 = *(_QWORD *)(v35 + 16);
          if ( !v43 )
          {
            v74[0] = 0;
            v49 = *(_DWORD *)(v35 + 8);
            ++*(_QWORD *)v35;
            v50 = (v49 >> 1) + 1;
LABEL_62:
            v51 = 3 * v43;
            goto LABEL_63;
          }
          v21 = v43 - 1;
        }
        v22 = v21 & (((unsigned int)v73 >> 9) ^ ((unsigned int)v73 >> 4));
        v23 = (_QWORD *)(v20 + 16LL * v22);
        v24 = *v23;
        if ( *v23 == v73 )
        {
LABEL_20:
          v25 = v23[1];
          goto LABEL_21;
        }
        v54 = 1;
        v55 = 0;
        while ( v24 != -4096 )
        {
          if ( !v55 && v24 == -8192 )
            v55 = v23;
          v22 = v21 & (v54 + v22);
          v23 = (_QWORD *)(v20 + 16LL * v22);
          v24 = *v23;
          if ( v73 == *v23 )
            goto LABEL_20;
          ++v54;
        }
        v51 = 24;
        v43 = 8;
        if ( !v55 )
          v55 = v23;
        v74[0] = v55;
        v49 = *(_DWORD *)(v35 + 8);
        ++*(_QWORD *)v35;
        v50 = (v49 >> 1) + 1;
        if ( !v36 )
        {
          v43 = *(_DWORD *)(v35 + 24);
          goto LABEL_62;
        }
LABEL_63:
        if ( v51 <= 4 * v50 )
        {
          v43 *= 2;
LABEL_69:
          sub_FADF20(v35, v43);
          sub_F9DCC0(v35, &v73, v74);
          v49 = *(_DWORD *)(v35 + 8);
          goto LABEL_65;
        }
        if ( v43 - *(_DWORD *)(v35 + 12) - v50 <= v43 >> 3 )
          goto LABEL_69;
LABEL_65:
        *(_DWORD *)(v35 + 8) = (2 * (v49 >> 1) + 2) | v49 & 1;
        v52 = (__int64 *)v74[0];
        if ( *(_QWORD *)v74[0] != -4096 )
          --*(_DWORD *)(v35 + 12);
        v53 = v73;
        v52[1] = 0;
        *v52 = v53;
        v25 = 0;
LABEL_21:
        if ( v19 != v25 )
          return 0;
        if ( !v12 )
          BUG();
        v26 = *(_QWORD *)(v12 + 32);
        if ( !v26 )
          goto LABEL_124;
        v12 = 0;
        if ( *(_BYTE *)(v26 - 24) == 84 )
          v12 = v26 - 24;
      }
      while ( v13 != v12 );
    }
    return 1;
  }
  return v2;
}
