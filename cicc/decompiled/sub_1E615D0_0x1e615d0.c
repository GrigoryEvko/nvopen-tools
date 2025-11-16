// Function: sub_1E615D0
// Address: 0x1e615d0
//
__int64 __fastcall sub_1E615D0(__int64 *a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r15
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 result; // rax
  __int64 *v11; // r13
  __int64 *v12; // r12
  __int64 *v13; // r14
  __int64 v14; // r10
  unsigned int v15; // esi
  __int64 v16; // rdi
  __int64 v17; // r9
  unsigned int v18; // eax
  int v19; // eax
  __int64 v20; // r13
  int v21; // edx
  __int64 v22; // rdi
  int v23; // r9d
  unsigned int v24; // eax
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned int v28; // r8d
  int v29; // esi
  __int64 v30; // rdx
  int v31; // edi
  __int64 v32; // r14
  __int64 *v33; // r13
  unsigned int v34; // r12d
  __int64 v35; // rcx
  __int64 v36; // r9
  __int64 v37; // r11
  unsigned int v38; // edx
  __int64 *v39; // rdi
  __int64 v40; // r8
  unsigned int v41; // eax
  unsigned int v42; // esi
  int v43; // esi
  int v44; // esi
  __int64 v45; // r8
  int v46; // edx
  __int64 v47; // rcx
  __int64 *v48; // rax
  __int64 v49; // rdi
  int v50; // edi
  int v51; // esi
  int v52; // esi
  __int64 v53; // r8
  __int64 *v54; // r10
  int v55; // r12d
  __int64 v56; // rcx
  __int64 v57; // rdi
  int v58; // ecx
  int v59; // edi
  int v60; // r12d
  __int64 *v61; // r13
  unsigned int v62; // [rsp+10h] [rbp-90h]
  __int64 v64; // [rsp+28h] [rbp-78h]
  __int64 *v65; // [rsp+28h] [rbp-78h]
  __int64 v66; // [rsp+30h] [rbp-70h]
  int v67; // [rsp+30h] [rbp-70h]
  __int64 *v69; // [rsp+40h] [rbp-60h]
  unsigned int v70; // [rsp+48h] [rbp-58h]
  __int64 v71; // [rsp+48h] [rbp-58h]
  __int64 v72; // [rsp+58h] [rbp-48h] BYREF
  __int64 v73; // [rsp+60h] [rbp-40h] BYREF
  __int64 v74[7]; // [rsp+68h] [rbp-38h] BYREF

  v4 = *a1;
  v5 = (a1[1] - *a1) >> 3;
  v62 = v5;
  if ( (unsigned int)v5 > 1 )
  {
    v6 = 8;
    v7 = 8LL * (unsigned int)v5;
    while ( 1 )
    {
      v8 = *(_QWORD *)(v4 + v6);
      v6 += 8;
      v74[0] = v8;
      v9 = sub_1E60050((__int64)(a1 + 3), v74);
      v9[4] = *(_QWORD *)(*a1 + 8LL * *((unsigned int *)v9 + 3));
      if ( v6 == v7 )
        break;
      v4 = *a1;
    }
  }
  result = v62 - 1;
  if ( (unsigned int)result <= 1 )
    return result;
  v66 = 8 * result;
  v64 = (__int64)(a1 + 3);
  v70 = v62;
  do
  {
    v72 = *(_QWORD *)(*a1 + v66);
    v11 = sub_1E60050(v64, &v72);
    v12 = (__int64 *)v11[5];
    *((_DWORD *)v11 + 4) = *((_DWORD *)v11 + 3);
    v13 = &v12[*((unsigned int *)v11 + 12)];
    if ( v12 != v13 )
    {
      v69 = v11;
      while ( 1 )
      {
        v19 = *((_DWORD *)a1 + 12);
        if ( !v19 )
          goto LABEL_13;
        v20 = *v12;
        v21 = v19 - 1;
        v22 = a1[4];
        v23 = 1;
        v24 = (v19 - 1) & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4));
        v25 = *(_QWORD *)(v22 + 72LL * (v21 & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4))));
        if ( *v12 == v25 )
        {
LABEL_16:
          v26 = sub_1E5E8E0(a2, *v12);
          if ( !v26 || a3 <= *(_DWORD *)(v26 + 16) )
          {
            v27 = sub_1E601A0(a1, v20, v70);
            v28 = *((_DWORD *)a1 + 12);
            v73 = v27;
            if ( !v28 )
            {
              ++a1[3];
              goto LABEL_20;
            }
            v14 = a1[4];
            v15 = (v28 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
            v16 = v14 + 72LL * v15;
            v17 = *(_QWORD *)v16;
            if ( v27 == *(_QWORD *)v16 )
            {
              v18 = *(_DWORD *)(v16 + 16);
            }
            else
            {
              v58 = 1;
              v30 = 0;
              while ( v17 != -8 )
              {
                if ( v30 || v17 != -16 )
                  v16 = v30;
                v15 = (v28 - 1) & (v58 + v15);
                v61 = (__int64 *)(v14 + 72LL * v15);
                v17 = *v61;
                if ( v27 == *v61 )
                {
                  v18 = *((_DWORD *)v61 + 4);
                  goto LABEL_11;
                }
                v30 = v16;
                ++v58;
                v16 = v14 + 72LL * v15;
              }
              if ( !v30 )
                v30 = v16;
              v59 = *((_DWORD *)a1 + 10);
              ++a1[3];
              v31 = v59 + 1;
              if ( 4 * v31 >= 3 * v28 )
              {
LABEL_20:
                v29 = 2 * v28;
              }
              else
              {
                if ( v28 - *((_DWORD *)a1 + 11) - v31 > v28 >> 3 )
                  goto LABEL_22;
                v29 = v28;
              }
              sub_1E5FC50(v64, v29);
              sub_1E5FA10(v64, &v73, v74);
              v30 = v74[0];
              v27 = v73;
              v31 = *((_DWORD *)a1 + 10) + 1;
LABEL_22:
              *((_DWORD *)a1 + 10) = v31;
              if ( *(_QWORD *)v30 != -8 )
                --*((_DWORD *)a1 + 11);
              *(_QWORD *)v30 = v27;
              *(_QWORD *)(v30 + 40) = v30 + 56;
              *(_QWORD *)(v30 + 48) = 0x200000000LL;
              v18 = 0;
              *(_OWORD *)(v30 + 8) = 0;
              *(_OWORD *)(v30 + 24) = 0;
              *(_OWORD *)(v30 + 56) = 0;
            }
LABEL_11:
            if ( *((_DWORD *)v69 + 4) > v18 )
              *((_DWORD *)v69 + 4) = v18;
          }
LABEL_13:
          if ( v13 == ++v12 )
            break;
        }
        else
        {
          while ( v25 != -8 )
          {
            v24 = v21 & (v23 + v24);
            v25 = *(_QWORD *)(v22 + 72LL * v24);
            if ( v20 == v25 )
              goto LABEL_16;
            ++v23;
          }
          if ( v13 == ++v12 )
            break;
        }
      }
    }
    result = --v70;
    v66 -= 8;
  }
  while ( v70 != 2 );
  v32 = v64;
  if ( v62 <= 2 )
    return result;
  v71 = 16;
  do
  {
    v73 = *(_QWORD *)(*a1 + v71);
    v33 = sub_1E60050(v32, &v73);
    v34 = *((_DWORD *)sub_1E60050(v32, (__int64 *)(*a1 + 8LL * *((unsigned int *)v33 + 4))) + 2);
    v74[0] = v33[4];
LABEL_37:
    v42 = *((_DWORD *)a1 + 12);
    if ( !v42 )
    {
      ++a1[3];
LABEL_39:
      sub_1E5FC50(v32, 2 * v42);
      v43 = *((_DWORD *)a1 + 12);
      if ( v43 )
      {
        v37 = v74[0];
        v44 = v43 - 1;
        v45 = a1[4];
        v46 = *((_DWORD *)a1 + 10) + 1;
        LODWORD(v47) = v44 & ((LODWORD(v74[0]) >> 9) ^ (LODWORD(v74[0]) >> 4));
        v48 = (__int64 *)(v45 + 72LL * (unsigned int)v47);
        v49 = *v48;
        if ( *v48 != v74[0] )
        {
          v60 = 1;
          v54 = 0;
          while ( v49 != -8 )
          {
            if ( v49 == -16 && !v54 )
              v54 = v48;
            v47 = v44 & (unsigned int)(v47 + v60);
            v48 = (__int64 *)(v45 + 72 * v47);
            v49 = *v48;
            if ( v74[0] == *v48 )
              goto LABEL_41;
            ++v60;
          }
          goto LABEL_55;
        }
        goto LABEL_41;
      }
LABEL_89:
      ++*((_DWORD *)a1 + 10);
      BUG();
    }
    v35 = v74[0];
    v36 = a1[4];
    v37 = v74[0];
    v38 = (v42 - 1) & ((LODWORD(v74[0]) >> 9) ^ (LODWORD(v74[0]) >> 4));
    v39 = (__int64 *)(v36 + 72LL * v38);
    v40 = *v39;
    if ( v74[0] == *v39 )
    {
      v41 = *((_DWORD *)v39 + 2);
      goto LABEL_35;
    }
    v67 = 1;
    v48 = 0;
    while ( v40 != -8 )
    {
      if ( v48 || v40 != -16 )
        v39 = v48;
      v38 = (v42 - 1) & (v67 + v38);
      v65 = (__int64 *)(v36 + 72LL * v38);
      v40 = *v65;
      if ( v74[0] == *v65 )
      {
        v41 = *((_DWORD *)v65 + 2);
LABEL_35:
        if ( v34 >= v41 )
          goto LABEL_44;
        v74[0] = sub_1E60050(v32, v74)[4];
        goto LABEL_37;
      }
      ++v67;
      v48 = v39;
      v39 = (__int64 *)(v36 + 72LL * v38);
    }
    if ( !v48 )
      v48 = v39;
    v50 = *((_DWORD *)a1 + 10);
    ++a1[3];
    v46 = v50 + 1;
    if ( 4 * (v50 + 1) >= 3 * v42 )
      goto LABEL_39;
    if ( v42 - *((_DWORD *)a1 + 11) - v46 > v42 >> 3 )
      goto LABEL_41;
    sub_1E5FC50(v32, v42);
    v51 = *((_DWORD *)a1 + 12);
    if ( !v51 )
      goto LABEL_89;
    v37 = v74[0];
    v52 = v51 - 1;
    v53 = a1[4];
    v54 = 0;
    v55 = 1;
    v46 = *((_DWORD *)a1 + 10) + 1;
    LODWORD(v56) = v52 & ((LODWORD(v74[0]) >> 9) ^ (LODWORD(v74[0]) >> 4));
    v48 = (__int64 *)(v53 + 72LL * (unsigned int)v56);
    v57 = *v48;
    if ( *v48 != v74[0] )
    {
      while ( v57 != -8 )
      {
        if ( !v54 && v57 == -16 )
          v54 = v48;
        v56 = v52 & (unsigned int)(v56 + v55);
        v48 = (__int64 *)(v53 + 72 * v56);
        v57 = *v48;
        if ( v74[0] == *v48 )
          goto LABEL_41;
        ++v55;
      }
LABEL_55:
      if ( v54 )
        v48 = v54;
    }
LABEL_41:
    *((_DWORD *)a1 + 10) = v46;
    if ( *v48 != -8 )
      --*((_DWORD *)a1 + 11);
    *v48 = v37;
    v48[5] = (__int64)(v48 + 7);
    v48[6] = 0x200000000LL;
    *(_OWORD *)(v48 + 1) = 0;
    *(_OWORD *)(v48 + 3) = 0;
    *(_OWORD *)(v48 + 7) = 0;
    v35 = v74[0];
LABEL_44:
    v71 += 8;
    result = v71;
    v33[4] = v35;
  }
  while ( 8LL * v62 != v71 );
  return result;
}
