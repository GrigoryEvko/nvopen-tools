// Function: sub_10754A0
// Address: 0x10754a0
//
__int64 __fastcall sub_10754A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // r13d
  __int64 *v9; // r12
  __int64 v10; // r8
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 *v13; // r12
  __int64 *v14; // r15
  __int64 v15; // r8
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 *v18; // r8
  __int64 result; // rax
  __int64 v20; // r13
  int v21; // r11d
  __int64 v22; // r9
  __int64 *v23; // rdx
  unsigned int v24; // edi
  __int64 *v25; // rax
  __int64 v26; // rcx
  __int64 *v27; // rax
  __int64 v28; // r13
  __int64 v29; // r15
  unsigned int v30; // esi
  __int64 v31; // r12
  int v32; // esi
  int v33; // esi
  __int64 v34; // r9
  unsigned int v35; // ecx
  int v36; // eax
  __int64 v37; // rdi
  int v38; // eax
  int v39; // ecx
  int v40; // ecx
  __int64 *v41; // r9
  unsigned int v42; // r13d
  __int64 v43; // rdi
  int v44; // r11d
  __int64 v45; // rsi
  int v46; // r13d
  __int64 *v47; // r10
  __int64 v48; // [rsp+8h] [rbp-48h]
  __int64 v49; // [rsp+10h] [rbp-40h]
  __int64 v50; // [rsp+10h] [rbp-40h]
  int v51; // [rsp+10h] [rbp-40h]
  __int64 *v52; // [rsp+18h] [rbp-38h]
  __int64 *v53; // [rsp+18h] [rbp-38h]
  __int64 *v54; // [rsp+18h] [rbp-38h]

  v7 = 0;
  v9 = *(__int64 **)(a2 + 40);
  v10 = (__int64)&v9[*(unsigned int *)(a2 + 48)];
  if ( v9 != (__int64 *)v10 )
  {
    do
    {
      while ( 1 )
      {
        v11 = *v9;
        if ( (*(_BYTE *)(*v9 + 48) & 0x20) == 0 )
          break;
        if ( (__int64 *)v10 == ++v9 )
          goto LABEL_8;
      }
      v12 = *(unsigned int *)(a1 + 264);
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 268) )
      {
        v50 = v10;
        sub_C8D5F0(a1 + 256, (const void *)(a1 + 272), v12 + 1, 8u, v10, a6);
        v12 = *(unsigned int *)(a1 + 264);
        v10 = v50;
      }
      ++v9;
      *(_QWORD *)(*(_QWORD *)(a1 + 256) + 8 * v12) = v11;
      ++*(_DWORD *)(a1 + 264);
      *(_DWORD *)(v11 + 172) = v7++;
    }
    while ( (__int64 *)v10 != v9 );
LABEL_8:
    v13 = *(__int64 **)(a2 + 40);
    v14 = &v13[*(unsigned int *)(a2 + 48)];
    if ( v14 != v13 )
    {
      v15 = v7;
      do
      {
        while ( 1 )
        {
          v16 = *v13;
          if ( (*(_BYTE *)(*v13 + 48) & 0x20) != 0 )
            break;
          if ( v14 == ++v13 )
            goto LABEL_15;
        }
        v17 = *(unsigned int *)(a1 + 264);
        if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 268) )
        {
          v51 = v15;
          sub_C8D5F0(a1 + 256, (const void *)(a1 + 272), v17 + 1, 8u, v15, a6);
          v17 = *(unsigned int *)(a1 + 264);
          LODWORD(v15) = v51;
        }
        ++v13;
        *(_QWORD *)(*(_QWORD *)(a1 + 256) + 8 * v17) = v16;
        ++*(_DWORD *)(a1 + 264);
        *(_DWORD *)(v16 + 172) = v15;
        v15 = (unsigned int)(v15 + 1);
      }
      while ( v14 != v13 );
    }
  }
LABEL_15:
  v18 = *(__int64 **)(a1 + 256);
  result = (__int64)&v18[*(unsigned int *)(a1 + 264)];
  v49 = result;
  if ( (__int64 *)result != v18 )
  {
    v48 = a1 + 224;
    v20 = 0;
    while ( 1 )
    {
      v29 = *v18;
      v30 = *(_DWORD *)(a1 + 248);
      v31 = -(1LL << *(_BYTE *)(*v18 + 32)) & (v20 + (1LL << *(_BYTE *)(*v18 + 32)) - 1);
      if ( !v30 )
        break;
      v21 = 1;
      v22 = *(_QWORD *)(a1 + 232);
      v23 = 0;
      v24 = (v30 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v25 = (__int64 *)(v22 + 16LL * v24);
      v26 = *v25;
      if ( v29 != *v25 )
      {
        while ( v26 != -4096 )
        {
          if ( v26 == -8192 && !v23 )
            v23 = v25;
          v24 = (v30 - 1) & (v21 + v24);
          v25 = (__int64 *)(v22 + 16LL * v24);
          v26 = *v25;
          if ( v29 == *v25 )
            goto LABEL_18;
          ++v21;
        }
        if ( !v23 )
          v23 = v25;
        v38 = *(_DWORD *)(a1 + 240);
        ++*(_QWORD *)(a1 + 224);
        v36 = v38 + 1;
        if ( 4 * v36 < 3 * v30 )
        {
          if ( v30 - *(_DWORD *)(a1 + 244) - v36 <= v30 >> 3 )
          {
            v54 = v18;
            sub_10752C0(v48, v30);
            v39 = *(_DWORD *)(a1 + 248);
            if ( !v39 )
            {
LABEL_59:
              ++*(_DWORD *)(a1 + 240);
              BUG();
            }
            v40 = v39 - 1;
            v41 = 0;
            v18 = v54;
            v42 = v40 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
            v43 = *(_QWORD *)(a1 + 232);
            v44 = 1;
            v36 = *(_DWORD *)(a1 + 240) + 1;
            v23 = (__int64 *)(v43 + 16LL * v42);
            v45 = *v23;
            if ( v29 != *v23 )
            {
              while ( v45 != -4096 )
              {
                if ( v45 == -8192 && !v41 )
                  v41 = v23;
                v42 = v40 & (v44 + v42);
                v23 = (__int64 *)(v43 + 16LL * v42);
                v45 = *v23;
                if ( v29 == *v23 )
                  goto LABEL_24;
                ++v44;
              }
              if ( v41 )
                v23 = v41;
            }
          }
          goto LABEL_24;
        }
LABEL_22:
        v53 = v18;
        sub_10752C0(v48, 2 * v30);
        v32 = *(_DWORD *)(a1 + 248);
        if ( !v32 )
          goto LABEL_59;
        v33 = v32 - 1;
        v34 = *(_QWORD *)(a1 + 232);
        v18 = v53;
        v35 = v33 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v36 = *(_DWORD *)(a1 + 240) + 1;
        v23 = (__int64 *)(v34 + 16LL * v35);
        v37 = *v23;
        if ( v29 != *v23 )
        {
          v46 = 1;
          v47 = 0;
          while ( v37 != -4096 )
          {
            if ( v37 == -8192 && !v47 )
              v47 = v23;
            v35 = v33 & (v46 + v35);
            v23 = (__int64 *)(v34 + 16LL * v35);
            v37 = *v23;
            if ( v29 == *v23 )
              goto LABEL_24;
            ++v46;
          }
          if ( v47 )
            v23 = v47;
        }
LABEL_24:
        *(_DWORD *)(a1 + 240) = v36;
        if ( *v23 != -4096 )
          --*(_DWORD *)(a1 + 244);
        *v23 = v29;
        v27 = v23 + 1;
        v23[1] = 0;
        goto LABEL_19;
      }
LABEL_18:
      v27 = v25 + 1;
LABEL_19:
      *v27 = v31;
      v52 = v18;
      v28 = sub_E5CAC0((__int64 *)a2, v29);
      result = sub_1070FC0(a1, (__int64 *)a2, v29);
      v20 = v31 + result + v28;
      v18 = v52 + 1;
      if ( (__int64 *)v49 == v52 + 1 )
        return result;
    }
    ++*(_QWORD *)(a1 + 224);
    goto LABEL_22;
  }
  return result;
}
