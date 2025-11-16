// Function: sub_1E716C0
// Address: 0x1e716c0
//
__int64 __fastcall sub_1E716C0(__int64 a1)
{
  int v1; // r8d
  int v2; // r9d
  __int64 v3; // rdx
  int *v4; // rax
  int v6; // r10d
  __int64 v7; // r15
  unsigned int v8; // ebx
  unsigned __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rax
  unsigned __int64 v12; // r12
  int v13; // eax
  __int64 v14; // r12
  __int64 *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rcx
  unsigned int v21; // esi
  __int64 *v22; // rdx
  __int64 v23; // r10
  __int64 v24; // rax
  char v25; // al
  unsigned int v26; // r10d
  unsigned int v27; // edx
  __int64 v28; // rsi
  __int64 v29; // rax
  _DWORD *v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // r15
  __int64 v33; // r13
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // rcx
  unsigned __int64 i; // rax
  __int64 v38; // rdi
  __int64 v39; // rcx
  unsigned int v40; // esi
  __int64 *v41; // rdx
  __int64 v42; // r9
  unsigned __int64 v43; // r12
  __int64 *v44; // rcx
  __int64 v45; // rsi
  __int64 v46; // rax
  char v47; // dl
  unsigned int v48; // eax
  unsigned int v49; // ecx
  unsigned int v50; // eax
  unsigned int v51; // eax
  int v53; // edx
  unsigned int v54; // r13d
  __int64 v55; // rcx
  __int64 v56; // rax
  _QWORD *v57; // rsi
  _QWORD *v58; // rax
  __int64 v59; // rdx
  int v60; // r8d
  int v61; // edx
  __int64 v62; // [rsp+8h] [rbp-68h]
  unsigned int v63; // [rsp+8h] [rbp-68h]
  int *v64; // [rsp+10h] [rbp-60h]
  unsigned int v65; // [rsp+18h] [rbp-58h]
  unsigned int v66; // [rsp+1Ch] [rbp-54h]
  __int64 v67; // [rsp+20h] [rbp-50h]
  int v68; // [rsp+20h] [rbp-50h]
  unsigned int v69; // [rsp+28h] [rbp-48h]
  __int64 v70; // [rsp+30h] [rbp-40h]
  int *v71; // [rsp+38h] [rbp-38h]

  if ( !sub_1DD6970(*(_QWORD *)(a1 + 920), *(_QWORD *)(a1 + 920)) )
    return 0;
  v3 = *(_QWORD *)(a1 + 2824);
  v4 = *(int **)(v3 + 104);
  v64 = &v4[2 * *(unsigned int *)(v3 + 112)];
  if ( v4 == v64 )
    return 0;
  v71 = *(int **)(v3 + 104);
  v66 = 0;
  v70 = a1 + 344;
  while ( 2 )
  {
    while ( 1 )
    {
      v6 = *v71;
      if ( *v71 >= 0 )
        break;
      v7 = *(_QWORD *)(a1 + 2112);
      v8 = v6 & 0x7FFFFFFF;
      v9 = *(unsigned int *)(v7 + 408);
      v62 = v6 & 0x7FFFFFFF;
      if ( (v6 & 0x7FFFFFFFu) < (unsigned int)v9 )
      {
        v10 = *(_QWORD *)(*(_QWORD *)(v7 + 400) + 8LL * v8);
        if ( v10 )
          goto LABEL_8;
      }
      v54 = v8 + 1;
      if ( (unsigned int)v9 < v8 + 1 )
      {
        v56 = v54;
        if ( v54 >= v9 )
        {
          if ( v54 > v9 )
          {
            if ( v54 > (unsigned __int64)*(unsigned int *)(v7 + 412) )
            {
              v68 = *v71;
              sub_16CD150(v7 + 400, (const void *)(v7 + 416), v54, 8, v1, v2);
              v9 = *(unsigned int *)(v7 + 408);
              v6 = v68;
              v56 = v54;
            }
            v55 = *(_QWORD *)(v7 + 400);
            v57 = (_QWORD *)(v55 + 8 * v56);
            v58 = (_QWORD *)(v55 + 8 * v9);
            v59 = *(_QWORD *)(v7 + 416);
            if ( v57 != v58 )
            {
              do
                *v58++ = v59;
              while ( v57 != v58 );
              v55 = *(_QWORD *)(v7 + 400);
            }
            *(_DWORD *)(v7 + 408) = v54;
            goto LABEL_65;
          }
        }
        else
        {
          *(_DWORD *)(v7 + 408) = v54;
        }
      }
      v55 = *(_QWORD *)(v7 + 400);
LABEL_65:
      *(_QWORD *)(v55 + 8 * v62) = sub_1DBA290(v6);
      v10 = *(_QWORD *)(*(_QWORD *)(v7 + 400) + 8 * v62);
      sub_1DBB110((_QWORD *)v7, v10);
      v7 = *(_QWORD *)(a1 + 2112);
LABEL_8:
      v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 272) + 392LL)
                      + 16LL * *(unsigned int *)(*(_QWORD *)(a1 + 920) + 48LL)
                      + 8);
      v12 = v11 & 0xFFFFFFFFFFFFFFF8LL;
      v13 = (v11 >> 1) & 3;
      if ( v13 )
        v14 = (2LL * (v13 - 1)) | v12;
      else
        v14 = *(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL | 6;
      v15 = (__int64 *)sub_1DB3C70((__int64 *)v10, v14);
      if ( v15 == (__int64 *)(*(_QWORD *)v10 + 24LL * *(unsigned int *)(v10 + 8)) )
        break;
      if ( (*(_DWORD *)((*v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v15 >> 1) & 3) > (*(_DWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                              | (unsigned int)(v14 >> 1)
                                                                                              & 3) )
        break;
      v16 = v15[2];
      if ( !v16 )
        break;
      v17 = 0;
      v18 = *(_QWORD *)(v16 + 8) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v18 )
        v17 = *(_QWORD *)(v18 + 16);
      v19 = *(unsigned int *)(a1 + 976);
      if ( !(_DWORD)v19 )
        break;
      v20 = *(_QWORD *)(a1 + 960);
      v21 = (v19 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v22 = (__int64 *)(v20 + 16LL * v21);
      v23 = *v22;
      if ( v17 != *v22 )
      {
        v61 = 1;
        while ( v23 != -8 )
        {
          v1 = v61 + 1;
          v21 = (v19 - 1) & (v61 + v21);
          v22 = (__int64 *)(v20 + 16LL * v21);
          v23 = *v22;
          if ( v17 == *v22 )
            goto LABEL_17;
          v61 = v1;
        }
        break;
      }
LABEL_17:
      if ( v22 == (__int64 *)(v20 + 16 * v19) )
        break;
      v24 = v22[1];
      v67 = v24;
      if ( !v24 )
        break;
      v25 = *(_BYTE *)(v24 + 236);
      if ( (v25 & 2) == 0 )
      {
        sub_1F01F70(v67);
        v25 = *(_BYTE *)(v67 + 236);
      }
      v65 = *(_DWORD *)(v67 + 244);
      if ( (v25 & 1) == 0 )
        sub_1F01DD0(v67);
      v26 = *(_DWORD *)(a1 + 2328);
      v69 = *(_DWORD *)(v67 + 240) + *(unsigned __int16 *)(v67 + 226);
      v27 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 2528) + v62);
      if ( v27 >= v26 )
        break;
      v28 = *(_QWORD *)(a1 + 2320);
      while ( 1 )
      {
        v29 = v27;
        v30 = (_DWORD *)(v28 + 24LL * v27);
        if ( v8 == (*v30 & 0x7FFFFFFF) )
        {
          v31 = (unsigned int)v30[4];
          if ( (_DWORD)v31 != -1 && *(_DWORD *)(v28 + 24 * v31 + 20) == -1 )
            break;
        }
        v27 += 256;
        if ( v26 <= v27 )
          goto LABEL_4;
      }
      if ( v27 == -1 )
        break;
      v32 = v10;
      do
      {
        v33 = 24 * v29;
        v34 = v28 + 24 * v29;
        v35 = *(_QWORD *)(v34 + 8);
        if ( v35 == v70 )
          goto LABEL_51;
        v36 = *(_QWORD *)(*(_QWORD *)(a1 + 2112) + 272LL);
        for ( i = *(_QWORD *)(v35 + 8); (*(_BYTE *)(i + 46) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
          ;
        v38 = *(_QWORD *)(v36 + 368);
        v39 = *(unsigned int *)(v36 + 384);
        if ( (_DWORD)v39 )
        {
          v40 = (v39 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
          v41 = (__int64 *)(v38 + 16LL * v40);
          v42 = *v41;
          if ( *v41 == i )
            goto LABEL_36;
          v53 = 1;
          while ( v42 != -8 )
          {
            v60 = v53 + 1;
            v40 = (v39 - 1) & (v53 + v40);
            v41 = (__int64 *)(v38 + 16LL * v40);
            v42 = *v41;
            if ( i == *v41 )
              goto LABEL_36;
            v53 = v60;
          }
        }
        v41 = (__int64 *)(v38 + 16 * v39);
LABEL_36:
        v43 = v41[1] & 0xFFFFFFFFFFFFFFF8LL;
        v44 = (__int64 *)sub_1DB3C70((__int64 *)v32, v43);
        v45 = *(_QWORD *)v32 + 24LL * *(unsigned int *)(v32 + 8);
        if ( v44 == (__int64 *)v45 )
          BUG();
        if ( (*(_DWORD *)((*v44 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v44 >> 1) & 3) > *(_DWORD *)(v43 + 24)
          || ((v46 = *(_QWORD *)(v44[2] + 8), v43 != (v44[1] & 0xFFFFFFFFFFFFFFF8LL)) || (__int64 *)v45 != v44 + 3)
          && v43 == v46 )
        {
          LOBYTE(v46) = MEMORY[8];
        }
        if ( (v46 & 6) != 0 )
          goto LABEL_50;
        v47 = *(_BYTE *)(v35 + 236);
        if ( (v47 & 1) != 0 )
        {
          v48 = *(_DWORD *)(v35 + 240);
          if ( v69 <= v48 )
            goto LABEL_58;
        }
        else
        {
          sub_1F01DD0(v35);
          v48 = *(_DWORD *)(v35 + 240);
          v47 = *(_BYTE *)(v35 + 236);
          if ( v69 <= v48 )
          {
LABEL_58:
            v49 = 0;
            if ( (v47 & 2) != 0 )
              goto LABEL_44;
            goto LABEL_59;
          }
          if ( (v47 & 1) == 0 )
          {
            sub_1F01DD0(v35);
            v48 = *(_DWORD *)(v35 + 240);
            v47 = *(_BYTE *)(v35 + 236);
          }
        }
        v49 = v69 - v48;
        if ( (v47 & 2) != 0 )
          goto LABEL_44;
LABEL_59:
        v63 = v49;
        sub_1F01F70(v35);
        v49 = v63;
LABEL_44:
        v50 = *(_DWORD *)(v35 + 244) + *(unsigned __int16 *)(v67 + 226);
        if ( v65 < v50 )
        {
          v51 = v50 - v65;
          if ( v51 > v49 )
            v51 = v49;
          if ( v66 >= v51 )
            v51 = v66;
          v66 = v51;
        }
LABEL_50:
        v28 = *(_QWORD *)(a1 + 2320);
        v34 = v28 + v33;
LABEL_51:
        v29 = *(unsigned int *)(v34 + 20);
      }
      while ( (_DWORD)v29 != -1 );
      v71 += 2;
      if ( v64 == v71 )
        return v66;
    }
LABEL_4:
    v71 += 2;
    if ( v64 != v71 )
      continue;
    return v66;
  }
}
