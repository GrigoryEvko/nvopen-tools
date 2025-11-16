// Function: sub_D6F970
// Address: 0xd6f970
//
__int64 __fastcall sub_D6F970(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rbx
  unsigned __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // r14
  unsigned int i; // r12d
  __int64 v9; // r8
  _QWORD *v10; // rdi
  _QWORD *v11; // rsi
  int v12; // ecx
  __int64 v13; // rsi
  int v14; // ecx
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // r9
  __int64 v18; // rsi
  unsigned int v19; // edi
  int v20; // r10d
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  _QWORD *v24; // rdx
  __int64 *v25; // rcx
  __int64 v26; // r14
  __int64 v27; // r12
  __int64 v28; // r11
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r10
  __int64 v32; // rax
  int v33; // eax
  __int64 v34; // rax
  __int64 v35; // rsi
  unsigned int v36; // ecx
  __int64 *v37; // rdx
  __int64 v38; // r8
  __int64 v39; // r8
  __int64 k; // rdi
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 *v46; // r14
  __int64 v47; // rdi
  __int64 v48; // r8
  __int64 v49; // rsi
  unsigned int v50; // ecx
  __int64 *v51; // rdx
  __int64 v52; // r10
  __int64 v53; // rbx
  __int64 v54; // rsi
  __int64 v55; // r13
  int v56; // eax
  __int64 v57; // rcx
  int v58; // edx
  unsigned int v59; // eax
  __int64 v60; // rsi
  int v61; // edi
  __int64 v62; // rax
  int j; // eax
  int v64; // edi
  int v65; // edx
  int v66; // r11d
  int v67; // edx
  int v68; // r9d
  __int64 v69; // [rsp+8h] [rbp-68h]
  __int64 v71; // [rsp+18h] [rbp-58h]
  unsigned int v72; // [rsp+24h] [rbp-4Ch]
  int v73; // [rsp+28h] [rbp-48h]
  __int64 *v74; // [rsp+28h] [rbp-48h]
  __int64 v75[7]; // [rsp+38h] [rbp-38h] BYREF

  v71 = *(_QWORD *)(a2 + 32);
  result = v71;
  v69 = v71 + 8LL * *(unsigned int *)(a2 + 40);
  if ( v69 == v71 )
    return result;
  do
  {
    v4 = *(_QWORD *)v71;
    v5 = *(_QWORD *)(*(_QWORD *)v71 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v5 == *(_QWORD *)v71 + 48LL )
    {
      v7 = *a1;
      goto LABEL_31;
    }
    if ( !v5 )
      BUG();
    v6 = v5 - 24;
    v7 = *a1;
    if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 <= 0xA )
    {
      v73 = sub_B46E30(v6);
      if ( v73 )
      {
        for ( i = 0; v73 != i; ++i )
        {
          v75[0] = sub_B46EC0(v6, i);
          v9 = v75[0];
          if ( *(_DWORD *)(a2 + 16) )
          {
            v56 = *(_DWORD *)(a2 + 24);
            v57 = *(_QWORD *)(a2 + 8);
            if ( v56 )
            {
              v58 = v56 - 1;
              v59 = (v56 - 1) & ((LODWORD(v75[0]) >> 9) ^ (LODWORD(v75[0]) >> 4));
              v60 = *(_QWORD *)(v57 + 8LL * v59);
              if ( v75[0] == v60 )
                continue;
              v61 = 1;
              while ( v60 != -4096 )
              {
                v59 = v58 & (v61 + v59);
                v60 = *(_QWORD *)(v57 + 8LL * v59);
                if ( v75[0] == v60 )
                  goto LABEL_30;
                ++v61;
              }
            }
          }
          else
          {
            v10 = *(_QWORD **)(a2 + 32);
            v11 = &v10[*(unsigned int *)(a2 + 40)];
            if ( v11 != sub_D67930(v10, (__int64)v11, v75) )
              continue;
          }
          v12 = *(_DWORD *)(v7 + 56);
          v13 = *(_QWORD *)(v7 + 40);
          if ( !v12 )
            continue;
          v14 = v12 - 1;
          v15 = v14 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v16 = (__int64 *)(v13 + 16LL * v15);
          v17 = *v16;
          if ( v9 != *v16 )
          {
            for ( j = 1; ; j = v64 )
            {
              if ( v17 == -4096 )
                goto LABEL_30;
              v64 = j + 1;
              v15 = v14 & (j + v15);
              v16 = (__int64 *)(v13 + 16LL * v15);
              v17 = *v16;
              if ( v9 == *v16 )
                break;
            }
          }
          v18 = v16[1];
          if ( !v18 )
            continue;
          v19 = 0;
          v20 = *(_DWORD *)(v18 + 4) & 0x7FFFFFF;
          if ( !v20 )
            goto LABEL_29;
          v72 = i;
          do
          {
            while ( 1 )
            {
              v21 = *(_QWORD *)(v18 - 8);
              v22 = 8LL * v19;
              v23 = 32LL * *(unsigned int *)(v18 + 76);
              v24 = (_QWORD *)(v21 + v23 + v22);
              if ( v4 == *v24 )
                break;
              if ( v20 == ++v19 )
                goto LABEL_28;
            }
            v25 = (__int64 *)(v21 + 32LL * v19);
            v26 = *v25;
            v27 = (*(_DWORD *)(v18 + 4) & 0x7FFFFFFu) - 1;
            v28 = *(_QWORD *)(v21 + 32 * v27);
            if ( v28 )
            {
              if ( v26 )
              {
                v29 = v25[1];
                *(_QWORD *)v25[2] = v29;
                if ( v29 )
                  *(_QWORD *)(v29 + 16) = v25[2];
              }
              *v25 = v28;
              v30 = *(_QWORD *)(v28 + 16);
              v25[1] = v30;
              if ( v30 )
                *(_QWORD *)(v30 + 16) = v25 + 1;
              v25[2] = v28 + 16;
              *(_QWORD *)(v28 + 16) = v25;
            }
            else
            {
              if ( !v26 )
                goto LABEL_24;
              v62 = v25[1];
              *(_QWORD *)v25[2] = v62;
              if ( v62 )
                *(_QWORD *)(v62 + 16) = v25[2];
              *v25 = 0;
            }
            v21 = *(_QWORD *)(v18 - 8);
            v23 = 32LL * *(unsigned int *)(v18 + 76);
            v24 = (_QWORD *)(v21 + v23 + v22);
LABEL_24:
            *v24 = *(_QWORD *)(v21 + v23 + 8 * v27);
            v31 = *(_QWORD *)(v18 - 8) + 32 * v27;
            if ( *(_QWORD *)v31 )
            {
              v32 = *(_QWORD *)(v31 + 8);
              **(_QWORD **)(v31 + 16) = v32;
              if ( v32 )
                *(_QWORD *)(v32 + 16) = *(_QWORD *)(v31 + 16);
            }
            *(_QWORD *)v31 = 0;
            *(_QWORD *)(*(_QWORD *)(v18 - 8) + 32LL * *(unsigned int *)(v18 + 76) + 8 * v27) = 0;
            v33 = *(_DWORD *)(v18 + 4);
            v20 = (v33 + 0x7FFFFFF) & 0x7FFFFFF;
            *(_DWORD *)(v18 + 4) = v20 | v33 & 0xF8000000;
          }
          while ( v20 != v19 );
LABEL_28:
          i = v72;
LABEL_29:
          sub_D6D630((__int64)a1, v18);
          v7 = *a1;
LABEL_30:
          ;
        }
      }
    }
LABEL_31:
    v34 = *(unsigned int *)(v7 + 88);
    v35 = *(_QWORD *)(v7 + 72);
    if ( (_DWORD)v34 )
    {
      v36 = (v34 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v37 = (__int64 *)(v35 + 16LL * v36);
      v38 = *v37;
      if ( v4 == *v37 )
      {
LABEL_33:
        if ( v37 != (__int64 *)(v35 + 16 * v34) )
        {
          v39 = v37[1];
          if ( v39 )
          {
            for ( k = *(_QWORD *)(v39 + 8); v39 != k; k = *(_QWORD *)(k + 8) )
            {
              if ( !k )
                BUG();
              v41 = 32LL * (*(_DWORD *)(k - 28) & 0x7FFFFFF);
              if ( (*(_BYTE *)(k - 25) & 0x40) != 0 )
              {
                v42 = *(_QWORD *)(k - 40);
                v43 = v42 + v41;
              }
              else
              {
                v43 = k - 32;
                v42 = k - 32 - v41;
              }
              for ( ; v42 != v43; v42 += 32 )
              {
                if ( *(_QWORD *)v42 )
                {
                  v44 = *(_QWORD *)(v42 + 8);
                  **(_QWORD **)(v42 + 16) = v44;
                  if ( v44 )
                    *(_QWORD *)(v44 + 16) = *(_QWORD *)(v42 + 16);
                }
                *(_QWORD *)v42 = 0;
              }
            }
          }
        }
      }
      else
      {
        v67 = 1;
        while ( v38 != -4096 )
        {
          v68 = v67 + 1;
          v36 = (v34 - 1) & (v67 + v36);
          v37 = (__int64 *)(v35 + 16LL * v36);
          v38 = *v37;
          if ( v4 == *v37 )
            goto LABEL_33;
          v67 = v68;
        }
      }
    }
    v71 += 8;
  }
  while ( v69 != v71 );
  v45 = *(_QWORD *)(a2 + 32);
  result = *(unsigned int *)(a2 + 40);
  if ( v45 + 8 * result != v45 )
  {
    v74 = (__int64 *)(v45 + 8 * result);
    v46 = *(__int64 **)(a2 + 32);
    do
    {
      v47 = *a1;
      v48 = *v46;
      result = *(unsigned int *)(*a1 + 88LL);
      v49 = *(_QWORD *)(*a1 + 72LL);
      if ( (_DWORD)result )
      {
        v50 = (result - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
        v51 = (__int64 *)(v49 + 16LL * v50);
        v52 = *v51;
        if ( v48 == *v51 )
        {
LABEL_50:
          result = v49 + 16 * result;
          if ( v51 != (__int64 *)result )
          {
            v53 = v51[1];
            if ( v53 )
            {
              v54 = *(_QWORD *)(v53 + 8);
              if ( v53 != v54 )
              {
                while ( 1 )
                {
                  v55 = *(_QWORD *)(v54 + 8);
                  sub_103E3E0(v47, v54 - 32);
                  result = sub_103CDC0(*a1, v54 - 32, 1);
                  if ( v53 == v55 )
                    break;
                  v47 = *a1;
                  v54 = v55;
                }
              }
            }
          }
        }
        else
        {
          v65 = 1;
          while ( v52 != -4096 )
          {
            v66 = v65 + 1;
            v50 = (result - 1) & (v65 + v50);
            v51 = (__int64 *)(v49 + 16LL * v50);
            v52 = *v51;
            if ( v48 == *v51 )
              goto LABEL_50;
            v65 = v66;
          }
        }
      }
      ++v46;
    }
    while ( v74 != v46 );
  }
  return result;
}
