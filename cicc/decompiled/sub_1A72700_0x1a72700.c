// Function: sub_1A72700
// Address: 0x1a72700
//
__int64 __fastcall sub_1A72700(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  unsigned int v7; // esi
  __int64 v8; // r8
  unsigned int v9; // r15d
  unsigned int v10; // r11d
  __int64 *v11; // r13
  __int64 v12; // rdx
  __int64 result; // rax
  __int64 v14; // rdx
  __int64 v15; // r12
  unsigned int v16; // esi
  int v17; // edx
  __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rax
  unsigned int v22; // r8d
  __int64 v23; // rax
  unsigned int v24; // esi
  __int64 v25; // r14
  __int64 v26; // r9
  __int64 v27; // r8
  __int64 v28; // rcx
  int v29; // r10d
  unsigned int v30; // edi
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // r8
  __int64 v35; // rax
  _QWORD *v36; // rax
  int v37; // eax
  int v38; // eax
  __int64 v39; // rsi
  _BYTE *v40; // rdi
  int v41; // edx
  int v42; // edx
  unsigned int v43; // esi
  __int64 v44; // rdi
  int v45; // r11d
  __int64 v46; // r8
  int v47; // r11d
  int v48; // r11d
  __int64 v49; // rdx
  int v50; // esi
  unsigned int v51; // r8d
  __int64 v52; // rdi
  int v53; // eax
  __int64 *v54; // rcx
  int v55; // eax
  int v56; // eax
  int v57; // edi
  int v58; // edi
  __int64 v59; // r8
  int v60; // edx
  __int64 *v61; // rsi
  unsigned int v62; // r15d
  __int64 v63; // rcx
  int v64; // edi
  int v65; // edi
  __int64 v66; // r9
  unsigned int v67; // edx
  __int64 v68; // r8
  int v69; // esi
  __int64 *v70; // rcx
  __int64 v71; // [rsp+10h] [rbp-C0h]
  __int64 v72; // [rsp+18h] [rbp-B8h]
  __int64 v73; // [rsp+20h] [rbp-B0h]
  __int64 v74; // [rsp+20h] [rbp-B0h]
  __int64 v75; // [rsp+20h] [rbp-B0h]
  __int64 v76; // [rsp+20h] [rbp-B0h]
  unsigned int v77; // [rsp+2Ch] [rbp-A4h]
  __int64 v78; // [rsp+60h] [rbp-70h] BYREF
  _BYTE *v79; // [rsp+68h] [rbp-68h] BYREF
  __int64 v80; // [rsp+70h] [rbp-60h]
  _BYTE v81[88]; // [rsp+78h] [rbp-58h] BYREF

  v5 = a1 + 416;
  v7 = *(_DWORD *)(a1 + 440);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 416);
    goto LABEL_77;
  }
  v8 = *(_QWORD *)(a1 + 424);
  v9 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
  v10 = (v7 - 1) & v9;
  v11 = (__int64 *)(v8 + ((unsigned __int64)v10 << 6));
  v12 = *v11;
  if ( *v11 == a3 )
    goto LABEL_3;
  v53 = 1;
  v54 = 0;
  while ( v12 != -8 )
  {
    if ( !v54 && v12 == -16 )
      v54 = v11;
    v10 = (v7 - 1) & (v53 + v10);
    v11 = (__int64 *)(v8 + ((unsigned __int64)v10 << 6));
    v12 = *v11;
    if ( *v11 == a3 )
      goto LABEL_3;
    ++v53;
  }
  v55 = *(_DWORD *)(a1 + 432);
  if ( v54 )
    v11 = v54;
  ++*(_QWORD *)(a1 + 416);
  v56 = v55 + 1;
  if ( 4 * v56 >= 3 * v7 )
  {
LABEL_77:
    sub_1A72280(v5, 2 * v7);
    v64 = *(_DWORD *)(a1 + 440);
    if ( v64 )
    {
      v65 = v64 - 1;
      v66 = *(_QWORD *)(a1 + 424);
      v67 = v65 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v56 = *(_DWORD *)(a1 + 432) + 1;
      v11 = (__int64 *)(v66 + ((unsigned __int64)v67 << 6));
      v68 = *v11;
      if ( *v11 != a3 )
      {
        v69 = 1;
        v70 = 0;
        while ( v68 != -8 )
        {
          if ( !v70 && v68 == -16 )
            v70 = v11;
          v67 = v65 & (v69 + v67);
          v11 = (__int64 *)(v66 + ((unsigned __int64)v67 << 6));
          v68 = *v11;
          if ( *v11 == a3 )
            goto LABEL_67;
          ++v69;
        }
        if ( v70 )
          v11 = v70;
      }
      goto LABEL_67;
    }
    goto LABEL_109;
  }
  if ( v7 - *(_DWORD *)(a1 + 436) - v56 <= v7 >> 3 )
  {
    sub_1A72280(v5, v7);
    v57 = *(_DWORD *)(a1 + 440);
    if ( v57 )
    {
      v58 = v57 - 1;
      v59 = *(_QWORD *)(a1 + 424);
      v60 = 1;
      v61 = 0;
      v62 = v58 & v9;
      v56 = *(_DWORD *)(a1 + 432) + 1;
      v11 = (__int64 *)(v59 + ((unsigned __int64)v62 << 6));
      v63 = *v11;
      if ( *v11 != a3 )
      {
        while ( v63 != -8 )
        {
          if ( v63 == -16 && !v61 )
            v61 = v11;
          v62 = v58 & (v60 + v62);
          v11 = (__int64 *)(v59 + ((unsigned __int64)v62 << 6));
          v63 = *v11;
          if ( *v11 == a3 )
            goto LABEL_67;
          ++v60;
        }
        if ( v61 )
          v11 = v61;
      }
      goto LABEL_67;
    }
LABEL_109:
    ++*(_DWORD *)(a1 + 432);
    BUG();
  }
LABEL_67:
  *(_DWORD *)(a1 + 432) = v56;
  if ( *v11 != -8 )
    --*(_DWORD *)(a1 + 436);
  *v11 = a3;
  v11[7] = 0;
  *(_OWORD *)(v11 + 1) = 0;
  *(_OWORD *)(v11 + 3) = 0;
  *(_OWORD *)(v11 + 5) = 0;
LABEL_3:
  result = sub_157F280(a3);
  v71 = v14;
  v15 = result;
  if ( v14 != result )
  {
    while ( 1 )
    {
      v77 = ((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4);
      v72 = (__int64)(v11 + 1);
      v16 = *(_DWORD *)(v15 + 20) & 0xFFFFFFF;
      if ( v16 )
        break;
LABEL_19:
      result = *(_QWORD *)(v15 + 32);
      if ( !result )
        BUG();
      v15 = 0;
      if ( *(_BYTE *)(result - 8) == 77 )
        v15 = result - 24;
      if ( v71 == v15 )
        return result;
    }
    while ( 1 )
    {
      v17 = 0;
      v18 = 24LL * *(unsigned int *)(v15 + 56);
      v19 = v18 + 8;
      while ( 1 )
      {
        v20 = v15 - 24LL * v16;
        if ( (*(_BYTE *)(v15 + 23) & 0x40) != 0 )
          v20 = *(_QWORD *)(v15 - 8);
        if ( a2 == *(_QWORD *)(v20 + v19) )
          break;
        ++v17;
        v19 += 8;
        if ( v16 == v17 )
          goto LABEL_19;
      }
      v21 = 0;
      while ( 1 )
      {
        v22 = v21;
        if ( a2 == *(_QWORD *)(v20 + v18 + 8 * v21 + 8) )
          break;
        if ( v16 == (_DWORD)++v21 )
        {
          v22 = -1;
          break;
        }
      }
      v23 = sub_15F5350(v15, v22, 0);
      v24 = *((_DWORD *)v11 + 8);
      v25 = v23;
      if ( !v24 )
        break;
      LODWORD(v26) = v24 - 1;
      v27 = v11[2];
      v28 = 0;
      v29 = 1;
      v30 = (v24 - 1) & v77;
      v31 = v27 + 16LL * v30;
      v32 = *(_QWORD *)v31;
      if ( v15 != *(_QWORD *)v31 )
      {
        while ( v32 != -8 )
        {
          if ( !v28 && v32 == -16 )
            v28 = v31;
          v30 = v26 & (v29 + v30);
          v31 = v27 + 16LL * v30;
          v32 = *(_QWORD *)v31;
          if ( v15 == *(_QWORD *)v31 )
            goto LABEL_15;
          ++v29;
        }
        if ( !v28 )
          v28 = v31;
        v37 = *((_DWORD *)v11 + 6);
        ++v11[1];
        v38 = v37 + 1;
        if ( 4 * v38 < 3 * v24 )
        {
          if ( v24 - *((_DWORD *)v11 + 7) - v38 <= v24 >> 3 )
          {
            sub_1A72540(v72, v24);
            v47 = *((_DWORD *)v11 + 8);
            if ( !v47 )
            {
LABEL_110:
              ++*((_DWORD *)v11 + 6);
              BUG();
            }
            v48 = v47 - 1;
            v26 = v11[2];
            v49 = 0;
            v50 = 1;
            v38 = *((_DWORD *)v11 + 6) + 1;
            v51 = v48 & v77;
            v28 = v26 + 16LL * (v48 & v77);
            v52 = *(_QWORD *)v28;
            if ( *(_QWORD *)v28 != v15 )
            {
              while ( v52 != -8 )
              {
                if ( v52 == -16 && !v49 )
                  v49 = v28;
                v51 = v48 & (v50 + v51);
                v28 = v26 + 16LL * v51;
                v52 = *(_QWORD *)v28;
                if ( v15 == *(_QWORD *)v28 )
                  goto LABEL_34;
                ++v50;
              }
              if ( v49 )
                v28 = v49;
            }
          }
          goto LABEL_34;
        }
LABEL_48:
        sub_1A72540(v72, 2 * v24);
        v41 = *((_DWORD *)v11 + 8);
        if ( !v41 )
          goto LABEL_110;
        v42 = v41 - 1;
        v26 = v11[2];
        v43 = v42 & v77;
        v38 = *((_DWORD *)v11 + 6) + 1;
        v28 = v26 + 16LL * (v42 & v77);
        v44 = *(_QWORD *)v28;
        if ( *(_QWORD *)v28 != v15 )
        {
          v45 = 1;
          v46 = 0;
          while ( v44 != -8 )
          {
            if ( !v46 && v44 == -16 )
              v46 = v28;
            v43 = v42 & (v45 + v43);
            v28 = v26 + 16LL * v43;
            v44 = *(_QWORD *)v28;
            if ( v15 == *(_QWORD *)v28 )
              goto LABEL_34;
            ++v45;
          }
          if ( v46 )
            v28 = v46;
        }
LABEL_34:
        *((_DWORD *)v11 + 6) = v38;
        if ( *(_QWORD *)v28 != -8 )
          --*((_DWORD *)v11 + 7);
        *(_QWORD *)v28 = v15;
        *(_DWORD *)(v28 + 8) = 0;
        v39 = v11[6];
        v78 = v15;
        v79 = v81;
        v80 = 0x200000000LL;
        if ( v39 == v11[7] )
        {
          v75 = v28;
          sub_1A6EA80(v11 + 5, (char *)v39, (__int64)&v78, v28);
          v40 = v79;
          v28 = v75;
        }
        else
        {
          v40 = v81;
          if ( v39 )
          {
            *(_QWORD *)v39 = v15;
            *(_QWORD *)(v39 + 8) = v39 + 24;
            *(_QWORD *)(v39 + 16) = 0x200000000LL;
            if ( (_DWORD)v80 )
            {
              v76 = v28;
              sub_1A6D150(v39 + 8, (__int64)&v79, (unsigned int)v80, v28, (int)&v79, v26);
              v39 = v11[6];
              v40 = v79;
              v28 = v76;
            }
            else
            {
              v39 = v11[6];
              v40 = v79;
            }
          }
          v11[6] = v39 + 56;
        }
        if ( v40 != v81 )
        {
          v73 = v28;
          _libc_free((unsigned __int64)v40);
          v28 = v73;
        }
        v33 = -1227133513 * (unsigned int)((v11[6] - v11[5]) >> 3) - 1;
        *(_DWORD *)(v28 + 8) = v33;
        goto LABEL_16;
      }
LABEL_15:
      v33 = *(unsigned int *)(v31 + 8);
LABEL_16:
      v34 = v11[5] + 56 * v33;
      v35 = *(unsigned int *)(v34 + 16);
      if ( (unsigned int)v35 >= *(_DWORD *)(v34 + 20) )
      {
        v74 = v34;
        sub_16CD150(v34 + 8, (const void *)(v34 + 24), 0, 16, v34, v26);
        v34 = v74;
        v35 = *(unsigned int *)(v74 + 16);
      }
      v36 = (_QWORD *)(*(_QWORD *)(v34 + 8) + 16 * v35);
      *v36 = a2;
      v36[1] = v25;
      ++*(_DWORD *)(v34 + 16);
      v16 = *(_DWORD *)(v15 + 20) & 0xFFFFFFF;
      if ( !v16 )
        goto LABEL_19;
    }
    ++v11[1];
    goto LABEL_48;
  }
  return result;
}
