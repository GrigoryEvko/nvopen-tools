// Function: sub_2EC7520
// Address: 0x2ec7520
//
__int64 __fastcall sub_2EC7520(__int64 a1)
{
  __int64 v2; // r8
  __int64 v3; // r9
  __int64 v4; // rdx
  int *v5; // rax
  int v6; // r10d
  __int64 v7; // r15
  unsigned int v8; // ebx
  unsigned __int64 v9; // rcx
  __int64 v10; // r13
  __int64 v11; // rdx
  signed __int64 v12; // r12
  __int64 *v13; // rdx
  __int64 v14; // rax
  int v15; // ecx
  __int64 v16; // rdi
  __int64 v17; // rsi
  int v18; // ecx
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r10
  char v22; // al
  unsigned int v23; // r10d
  unsigned int v24; // edx
  __int64 v25; // rsi
  __int64 v26; // rax
  _DWORD *v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // r15
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // rbx
  unsigned __int64 v33; // rsi
  __int64 v34; // rcx
  unsigned __int64 i; // rax
  __int64 j; // rdi
  __int16 v37; // dx
  __int64 v38; // rsi
  __int64 v39; // rdi
  unsigned int v40; // ecx
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
  unsigned int v54; // eax
  __int64 v55; // rsi
  __int64 v56; // rax
  unsigned __int64 v57; // rdx
  __int64 v58; // rax
  unsigned __int64 v59; // r13
  _QWORD *v60; // rdx
  _QWORD *v61; // rdi
  int v62; // r8d
  int v63; // eax
  unsigned int v64; // [rsp+8h] [rbp-68h]
  int *v65; // [rsp+10h] [rbp-60h]
  unsigned int v66; // [rsp+18h] [rbp-58h]
  unsigned int v67; // [rsp+1Ch] [rbp-54h]
  unsigned int v68; // [rsp+20h] [rbp-50h]
  __int64 v69; // [rsp+20h] [rbp-50h]
  __int64 v70; // [rsp+28h] [rbp-48h]
  int v71; // [rsp+28h] [rbp-48h]
  int *v72; // [rsp+38h] [rbp-38h]

  if ( !sub_2E322C0(*(_QWORD *)(a1 + 904), *(_QWORD *)(a1 + 904)) )
    return 0;
  v4 = *(_QWORD *)(a1 + 4528);
  v5 = *(int **)(v4 + 232);
  v65 = &v5[6 * *(unsigned int *)(v4 + 240)];
  if ( v5 == v65 )
    return 0;
  v72 = *(int **)(v4 + 232);
  v67 = 0;
  while ( 2 )
  {
    while ( 1 )
    {
      v6 = *v72;
      if ( *v72 >= 0 )
        break;
      v7 = *(_QWORD *)(a1 + 3464);
      v8 = v6 & 0x7FFFFFFF;
      v9 = *(unsigned int *)(v7 + 160);
      if ( (v6 & 0x7FFFFFFFu) >= (unsigned int)v9 || (v10 = *(_QWORD *)(*(_QWORD *)(v7 + 152) + 8LL * v8)) == 0 )
      {
        v54 = v8 + 1;
        if ( (unsigned int)v9 >= v8 + 1 || (v57 = v54, v54 == v9) )
        {
LABEL_67:
          v55 = *(_QWORD *)(v7 + 152);
        }
        else
        {
          if ( v54 < v9 )
          {
            *(_DWORD *)(v7 + 160) = v54;
            goto LABEL_67;
          }
          v58 = *(_QWORD *)(v7 + 168);
          v59 = v57 - v9;
          if ( v57 > *(unsigned int *)(v7 + 164) )
          {
            v69 = *(_QWORD *)(v7 + 168);
            v71 = *v72;
            sub_C8D5F0(v7 + 152, (const void *)(v7 + 168), v57, 8u, v2, v3);
            v9 = *(unsigned int *)(v7 + 160);
            v58 = v69;
            v6 = v71;
          }
          v55 = *(_QWORD *)(v7 + 152);
          v60 = (_QWORD *)(v55 + 8 * v9);
          v61 = &v60[v59];
          if ( v60 != v61 )
          {
            do
              *v60++ = v58;
            while ( v61 != v60 );
            LODWORD(v9) = *(_DWORD *)(v7 + 160);
            v55 = *(_QWORD *)(v7 + 152);
          }
          *(_DWORD *)(v7 + 160) = v59 + v9;
        }
        v56 = sub_2E10F30(v6);
        *(_QWORD *)(v55 + 8LL * v8) = v56;
        v10 = v56;
        sub_2E11E80((_QWORD *)v7, v56);
        v7 = *(_QWORD *)(a1 + 3464);
      }
      v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 32) + 152LL)
                      + 16LL * *(unsigned int *)(*(_QWORD *)(a1 + 904) + 24LL)
                      + 8);
      if ( ((v11 >> 1) & 3) != 0 )
        v12 = v11 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v11 >> 1) & 3) - 1));
      else
        v12 = *(_QWORD *)(v11 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
      v13 = (__int64 *)sub_2E09D00((__int64 *)v10, v12);
      if ( v13 == (__int64 *)(*(_QWORD *)v10 + 24LL * *(unsigned int *)(v10 + 8)) )
        break;
      if ( (*(_DWORD *)((*v13 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v13 >> 1) & 3)) > (*(_DWORD *)((v12 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                | (unsigned int)(v12 >> 1)
                                                                                                & 3) )
        break;
      v14 = v13[2];
      if ( !v14 )
        break;
      v15 = *(_DWORD *)(a1 + 960);
      v16 = *(_QWORD *)(a1 + 944);
      v17 = *(_QWORD *)((*(_QWORD *)(v14 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 16);
      if ( !v15 )
        break;
      v18 = v15 - 1;
      v19 = v18 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v20 = (__int64 *)(v16 + 16LL * v19);
      v21 = *v20;
      if ( v17 != *v20 )
      {
        v63 = 1;
        while ( v21 != -4096 )
        {
          v2 = (unsigned int)(v63 + 1);
          v19 = v18 & (v63 + v19);
          v20 = (__int64 *)(v16 + 16LL * v19);
          v21 = *v20;
          if ( v17 == *v20 )
            goto LABEL_15;
          v63 = v2;
        }
        break;
      }
LABEL_15:
      v70 = v20[1];
      if ( !v70 )
        break;
      v22 = *(_BYTE *)(v70 + 254);
      if ( (v22 & 2) == 0 )
      {
        sub_2F8F770(v70);
        v22 = *(_BYTE *)(v70 + 254);
      }
      v66 = *(_DWORD *)(v70 + 244);
      if ( (v22 & 1) == 0 )
        sub_2F8F5D0(v70);
      v23 = *(_DWORD *)(a1 + 3648);
      v68 = *(_DWORD *)(v70 + 240) + *(unsigned __int16 *)(v70 + 252);
      v24 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 3976) + v8);
      if ( v24 >= v23 )
        break;
      v25 = *(_QWORD *)(a1 + 3640);
      while ( 1 )
      {
        v26 = v24;
        v27 = (_DWORD *)(v25 + 40LL * v24);
        if ( v8 == (*v27 & 0x7FFFFFFF) )
        {
          v28 = (unsigned int)v27[8];
          if ( (_DWORD)v28 != -1 && *(_DWORD *)(v25 + 40 * v28 + 36) == -1 )
            break;
        }
        v24 += 256;
        if ( v23 <= v24 )
          goto LABEL_4;
      }
      if ( v24 == -1 )
        break;
      v29 = v10;
      do
      {
        v30 = 40 * v26;
        v31 = v25 + 40 * v26;
        v32 = *(_QWORD *)(v31 + 24);
        if ( v32 == a1 + 328 )
          goto LABEL_54;
        v33 = *(_QWORD *)v32;
        v34 = *(_QWORD *)(*(_QWORD *)(a1 + 3464) + 32LL);
        for ( i = *(_QWORD *)v32; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
          ;
        if ( (*(_DWORD *)(*(_QWORD *)v32 + 44LL) & 8) != 0 )
        {
          do
            v33 = *(_QWORD *)(v33 + 8);
          while ( (*(_BYTE *)(v33 + 44) & 8) != 0 );
        }
        for ( j = *(_QWORD *)(v33 + 8); j != i; i = *(_QWORD *)(i + 8) )
        {
          v37 = *(_WORD *)(i + 68);
          if ( (unsigned __int16)(v37 - 14) > 4u && v37 != 24 )
            break;
        }
        v38 = *(unsigned int *)(v34 + 144);
        v39 = *(_QWORD *)(v34 + 128);
        if ( (_DWORD)v38 )
        {
          v40 = (v38 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
          v41 = (__int64 *)(v39 + 16LL * v40);
          v42 = *v41;
          if ( i == *v41 )
            goto LABEL_39;
          v53 = 1;
          while ( v42 != -4096 )
          {
            v62 = v53 + 1;
            v40 = (v38 - 1) & (v53 + v40);
            v41 = (__int64 *)(v39 + 16LL * v40);
            v42 = *v41;
            if ( i == *v41 )
              goto LABEL_39;
            v53 = v62;
          }
        }
        v41 = (__int64 *)(v39 + 16 * v38);
LABEL_39:
        v43 = v41[1] & 0xFFFFFFFFFFFFFFF8LL;
        v44 = (__int64 *)sub_2E09D00((__int64 *)v29, v43);
        v45 = *(_QWORD *)v29 + 24LL * *(unsigned int *)(v29 + 8);
        if ( v44 == (__int64 *)v45 )
          BUG();
        if ( (*(_DWORD *)((*v44 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v44 >> 1) & 3) > *(_DWORD *)(v43 + 24)
          || ((v46 = *(_QWORD *)(v44[2] + 8), v43 != (v44[1] & 0xFFFFFFFFFFFFFFF8LL)) || (__int64 *)v45 != v44 + 3)
          && v43 == v46 )
        {
          LOBYTE(v46) = MEMORY[8];
        }
        if ( (v46 & 6) != 0 )
          goto LABEL_53;
        v47 = *(_BYTE *)(v32 + 254);
        if ( (v47 & 1) != 0 )
        {
          v48 = *(_DWORD *)(v32 + 240);
          if ( v68 <= v48 )
            goto LABEL_58;
        }
        else
        {
          sub_2F8F5D0(v32);
          v48 = *(_DWORD *)(v32 + 240);
          v47 = *(_BYTE *)(v32 + 254);
          if ( v68 <= v48 )
          {
LABEL_58:
            v49 = 0;
            if ( (v47 & 2) != 0 )
              goto LABEL_47;
            goto LABEL_59;
          }
          if ( (v47 & 1) == 0 )
          {
            sub_2F8F5D0(v32);
            v48 = *(_DWORD *)(v32 + 240);
            v47 = *(_BYTE *)(v32 + 254);
          }
        }
        v49 = v68 - v48;
        if ( (v47 & 2) != 0 )
          goto LABEL_47;
LABEL_59:
        v64 = v49;
        sub_2F8F770(v32);
        v49 = v64;
LABEL_47:
        v50 = *(_DWORD *)(v32 + 244) + *(unsigned __int16 *)(v70 + 252);
        if ( v66 < v50 )
        {
          v51 = v50 - v66;
          if ( v51 > v49 )
            v51 = v49;
          if ( v67 >= v51 )
            v51 = v67;
          v67 = v51;
        }
LABEL_53:
        v25 = *(_QWORD *)(a1 + 3640);
        v31 = v25 + v30;
LABEL_54:
        v26 = *(unsigned int *)(v31 + 36);
      }
      while ( (_DWORD)v26 != -1 );
      v72 += 6;
      if ( v65 == v72 )
        return v67;
    }
LABEL_4:
    v72 += 6;
    if ( v65 != v72 )
      continue;
    return v67;
  }
}
