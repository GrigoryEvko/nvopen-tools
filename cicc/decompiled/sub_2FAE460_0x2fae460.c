// Function: sub_2FAE460
// Address: 0x2fae460
//
void __fastcall sub_2FAE460(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  unsigned __int64 v8; // rax
  __int64 v9; // rbx
  _QWORD *v10; // rax
  __int64 i; // rdx
  __int64 v12; // rax
  __int64 v13; // rcx
  unsigned int v14; // edx
  __int64 v15; // rdx
  unsigned __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // r8
  unsigned __int64 v21; // r10
  int v22; // r14d
  __int64 v23; // rbx
  unsigned __int64 v24; // r11
  __int16 v25; // ax
  __int64 v26; // rdx
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  unsigned __int64 *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned __int64 v33; // r12
  __int64 v34; // rax
  unsigned int v35; // esi
  __int64 v36; // r12
  unsigned __int64 v37; // r12
  unsigned int v38; // edi
  __int64 *v39; // rax
  __int64 v40; // rcx
  int v41; // r10d
  __int64 *v42; // rdx
  int v43; // eax
  int v44; // eax
  __int64 v45; // rax
  int v46; // r15d
  int v47; // r15d
  __int64 v48; // r10
  unsigned int v49; // ecx
  int v50; // edi
  __int64 *v51; // rsi
  __int64 *v52; // r13
  __int64 v53; // rax
  unsigned __int64 v54; // rbx
  __int64 *v55; // r12
  char *v56; // rsi
  __int64 *v57; // rbx
  __int64 *v58; // rdi
  int v59; // r10d
  int v60; // r10d
  __int64 *v61; // rcx
  int v62; // esi
  unsigned int v63; // r15d
  __int64 v64; // rdi
  __int64 v65; // [rsp+0h] [rbp-60h]
  __int64 v66; // [rsp+8h] [rbp-58h]
  unsigned __int64 v67; // [rsp+10h] [rbp-50h]
  unsigned __int64 v68; // [rsp+10h] [rbp-50h]
  unsigned __int64 v69; // [rsp+10h] [rbp-50h]
  unsigned __int64 v70; // [rsp+18h] [rbp-48h]
  unsigned __int64 v71; // [rsp+18h] [rbp-48h]
  unsigned __int64 v72; // [rsp+18h] [rbp-48h]
  unsigned __int64 v73; // [rsp+20h] [rbp-40h]
  __int64 v74; // [rsp+28h] [rbp-38h]

  v6 = a2;
  *(_QWORD *)(a1 + 112) = a2;
  v8 = *(unsigned int *)(a1 + 160);
  v9 = (__int64)(*(_QWORD *)(a2 + 104) - *(_QWORD *)(a2 + 96)) >> 3;
  if ( (unsigned int)v9 != v8 )
  {
    if ( (unsigned int)v9 < v8 )
    {
      *(_DWORD *)(a1 + 160) = v9;
    }
    else
    {
      if ( (unsigned int)v9 > (unsigned __int64)*(unsigned int *)(a1 + 164) )
      {
        sub_C8D5F0(a1 + 152, (const void *)(a1 + 168), (unsigned int)v9, 0x10u, a5, a6);
        v8 = *(unsigned int *)(a1 + 160);
      }
      v10 = (_QWORD *)(*(_QWORD *)(a1 + 152) + 16 * v8);
      for ( i = *(_QWORD *)(a1 + 152) + 16LL * (unsigned int)v9; (_QWORD *)i != v10; v10 += 2 )
      {
        if ( v10 )
        {
          *v10 = 0;
          v10[1] = 0;
        }
      }
      *(_DWORD *)(a1 + 160) = v9;
      v6 = *(_QWORD *)(a1 + 112);
    }
  }
  v12 = *(_QWORD *)(v6 + 328);
  v13 = v6 + 320;
  if ( v12 != v6 + 320 )
  {
    v14 = 0;
    do
    {
      v12 = *(_QWORD *)(v12 + 8);
      ++v14;
    }
    while ( v12 != v13 );
    if ( *(_DWORD *)(a1 + 308) < v14 )
      sub_C8D5F0(a1 + 296, (const void *)(a1 + 312), v14, 0x10u, v14, a6);
  }
  v15 = *(_QWORD *)a1;
  *(_QWORD *)(a1 + 80) += 32LL;
  v74 = a1 + 96;
  v16 = (v15 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_QWORD *)(a1 + 8) >= v16 + 32 && v15 )
    *(_QWORD *)a1 = v16 + 32;
  else
    v16 = sub_9D1E70(a1, 32, 32, 3);
  *(_QWORD *)v16 = 0;
  *(_QWORD *)(v16 + 8) = 0;
  *(_QWORD *)(v16 + 16) = 0;
  *(_DWORD *)(v16 + 24) = 0;
  v17 = *(_QWORD *)(a1 + 96);
  *(_QWORD *)(v16 + 8) = v74;
  v17 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v16 = v17;
  *(_QWORD *)(v17 + 8) = v16;
  v18 = *(_QWORD *)(a1 + 96) & 7LL | v16;
  v19 = *(_QWORD *)(a1 + 112);
  *(_QWORD *)(a1 + 96) = v18;
  v20 = *(_QWORD *)(v19 + 328);
  v66 = v19 + 320;
  if ( v20 != v19 + 320 )
  {
    v21 = *(_QWORD *)(v19 + 328);
    v65 = a1 + 120;
    v22 = 0;
    while ( 1 )
    {
      v23 = *(_QWORD *)(v21 + 56);
      v24 = v21 + 48;
      v73 = v18 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v21 + 48 != v23 )
        break;
LABEL_28:
      v26 = *(_QWORD *)a1;
      *(_QWORD *)(a1 + 80) += 32LL;
      v22 += 16;
      v27 = (v26 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( *(_QWORD *)(a1 + 8) >= v27 + 32 && v26 )
      {
        *(_QWORD *)a1 = v27 + 32;
      }
      else
      {
        v71 = v21;
        v27 = sub_9D1E70(a1, 32, 32, 3);
        v21 = v71;
      }
      *(_QWORD *)v27 = 0;
      *(_QWORD *)(v27 + 8) = 0;
      *(_QWORD *)(v27 + 16) = 0;
      *(_DWORD *)(v27 + 24) = v22;
      v28 = *(_QWORD *)(a1 + 96);
      *(_QWORD *)(v27 + 8) = v74;
      v28 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v27 = v28;
      *(_QWORD *)(v28 + 8) = v27;
      *(_QWORD *)(a1 + 96) = *(_QWORD *)(a1 + 96) & 7LL | v27;
      *(_QWORD *)(*(_QWORD *)(a1 + 152) + 16LL * *(int *)(v21 + 24)) = v73;
      *(_QWORD *)(*(_QWORD *)(a1 + 152) + 16LL * *(int *)(v21 + 24) + 8) = *(_QWORD *)(a1 + 96) & 0xFFFFFFFFFFFFFFF8LL;
      v29 = *(unsigned int *)(a1 + 304);
      if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 308) )
      {
        v72 = v21;
        sub_C8D5F0(a1 + 296, (const void *)(a1 + 312), v29 + 1, 0x10u, v20, a6);
        v29 = *(unsigned int *)(a1 + 304);
        v21 = v72;
      }
      v30 = (unsigned __int64 *)(*(_QWORD *)(a1 + 296) + 16 * v29);
      v30[1] = v21;
      *v30 = v73;
      v31 = (unsigned int)(*(_DWORD *)(a1 + 304) + 1);
      *(_DWORD *)(a1 + 304) = v31;
      v21 = *(_QWORD *)(v21 + 8);
      if ( v66 == v21 )
        goto LABEL_63;
      v18 = *(_QWORD *)(a1 + 96);
    }
    v70 = v21;
    while ( 1 )
    {
      v25 = *(_WORD *)(v23 + 68);
      if ( (unsigned __int16)(v25 - 14) > 4u && v25 != 24 )
      {
        v32 = *(_QWORD *)a1;
        *(_QWORD *)(a1 + 80) += 32LL;
        v33 = (v32 + 7) & 0xFFFFFFFFFFFFFFF8LL;
        if ( *(_QWORD *)(a1 + 8) >= v33 + 32 && v32 )
        {
          *(_QWORD *)a1 = v33 + 32;
        }
        else
        {
          v67 = v24;
          v45 = sub_9D1E70(a1, 32, 32, 3);
          v24 = v67;
          v33 = v45;
        }
        v22 += 16;
        *(_QWORD *)(v33 + 16) = v23;
        *(_QWORD *)v33 = 0;
        *(_QWORD *)(v33 + 8) = 0;
        *(_DWORD *)(v33 + 24) = v22;
        v34 = *(_QWORD *)(a1 + 96);
        *(_QWORD *)(v33 + 8) = v74;
        v34 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v33 = v34;
        *(_QWORD *)(v34 + 8) = v33;
        v35 = *(_DWORD *)(a1 + 144);
        v36 = *(_QWORD *)(a1 + 96) & 7LL | v33;
        *(_QWORD *)(a1 + 96) = v36;
        v37 = v36 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v35 )
        {
          ++*(_QWORD *)(a1 + 120);
          goto LABEL_55;
        }
        a6 = v35 - 1;
        v20 = *(_QWORD *)(a1 + 128);
        v38 = a6 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v39 = (__int64 *)(v20 + 16LL * v38);
        v40 = *v39;
        if ( v23 != *v39 )
        {
          v41 = 1;
          v42 = 0;
          while ( v40 != -4096 )
          {
            if ( v42 || v40 != -8192 )
              v39 = v42;
            v38 = a6 & (v41 + v38);
            v40 = *(_QWORD *)(v20 + 16LL * v38);
            if ( v40 == v23 )
              goto LABEL_25;
            ++v41;
            v42 = v39;
            v39 = (__int64 *)(v20 + 16LL * v38);
          }
          if ( !v42 )
            v42 = v39;
          v43 = *(_DWORD *)(a1 + 136);
          ++*(_QWORD *)(a1 + 120);
          v44 = v43 + 1;
          if ( 4 * v44 >= 3 * v35 )
          {
LABEL_55:
            v68 = v24;
            sub_2E190F0(v65, 2 * v35);
            v46 = *(_DWORD *)(a1 + 144);
            if ( !v46 )
              goto LABEL_92;
            v47 = v46 - 1;
            v48 = *(_QWORD *)(a1 + 128);
            v24 = v68;
            v49 = v47 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
            v44 = *(_DWORD *)(a1 + 136) + 1;
            v42 = (__int64 *)(v48 + 16LL * v49);
            v20 = *v42;
            if ( *v42 != v23 )
            {
              v50 = 1;
              v51 = 0;
              while ( v20 != -4096 )
              {
                if ( !v51 && v20 == -8192 )
                  v51 = v42;
                a6 = (unsigned int)(v50 + 1);
                v49 = v47 & (v50 + v49);
                v42 = (__int64 *)(v48 + 16LL * v49);
                v20 = *v42;
                if ( v23 == *v42 )
                  goto LABEL_49;
                ++v50;
              }
              if ( v51 )
                v42 = v51;
            }
          }
          else if ( v35 - *(_DWORD *)(a1 + 140) - v44 <= v35 >> 3 )
          {
            v69 = v24;
            sub_2E190F0(v65, v35);
            v59 = *(_DWORD *)(a1 + 144);
            if ( !v59 )
            {
LABEL_92:
              ++*(_DWORD *)(a1 + 136);
              BUG();
            }
            v60 = v59 - 1;
            v61 = 0;
            v24 = v69;
            v62 = 1;
            v63 = v60 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
            a6 = *(_QWORD *)(a1 + 128);
            v44 = *(_DWORD *)(a1 + 136) + 1;
            v42 = (__int64 *)(a6 + 16LL * v63);
            v64 = *v42;
            if ( *v42 != v23 )
            {
              while ( v64 != -4096 )
              {
                if ( v64 == -8192 && !v61 )
                  v61 = v42;
                v20 = (unsigned int)(v62 + 1);
                v63 = v60 & (v62 + v63);
                v42 = (__int64 *)(a6 + 16LL * v63);
                v64 = *v42;
                if ( v23 == *v42 )
                  goto LABEL_49;
                ++v62;
              }
              if ( v61 )
                v42 = v61;
            }
          }
LABEL_49:
          *(_DWORD *)(a1 + 136) = v44;
          if ( *v42 != -4096 )
            --*(_DWORD *)(a1 + 140);
          *v42 = v23;
          v42[1] = v37;
        }
      }
LABEL_25:
      if ( (*(_BYTE *)v23 & 4) != 0 )
      {
        v23 = *(_QWORD *)(v23 + 8);
        if ( v24 == v23 )
          goto LABEL_27;
      }
      else
      {
        while ( (*(_BYTE *)(v23 + 44) & 8) != 0 )
          v23 = *(_QWORD *)(v23 + 8);
        v23 = *(_QWORD *)(v23 + 8);
        if ( v24 == v23 )
        {
LABEL_27:
          v21 = v70;
          goto LABEL_28;
        }
      }
    }
  }
  v31 = *(unsigned int *)(a1 + 304);
LABEL_63:
  v52 = *(__int64 **)(a1 + 296);
  v53 = 16 * v31;
  v54 = v53;
  v55 = (__int64 *)((char *)v52 + v53);
  if ( v52 != (__int64 *)((char *)v52 + v53) )
  {
    v56 = (char *)v52 + v53;
    _BitScanReverse64((unsigned __int64 *)&v53, v53 >> 4);
    sub_2E34B10(v52, v56, 2LL * (int)(63 - (v53 ^ 0x3F)));
    if ( v54 <= 0x100 )
    {
      sub_2FACCA0(v52, v55);
    }
    else
    {
      v57 = v52 + 32;
      sub_2FACCA0(v52, v52 + 32);
      if ( v55 != v52 + 32 )
      {
        do
        {
          v58 = v57;
          v57 += 2;
          sub_2FACC30(v58);
        }
        while ( v55 != v57 );
      }
    }
  }
}
