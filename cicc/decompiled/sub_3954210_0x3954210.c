// Function: sub_3954210
// Address: 0x3954210
//
__int64 __fastcall sub_3954210(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  char v6; // al
  __int64 v7; // rdx
  __int64 result; // rax
  unsigned int v9; // esi
  unsigned int v10; // r8d
  __int64 v11; // rdi
  int v12; // r10d
  unsigned int v13; // edx
  unsigned int v14; // r9d
  __int64 *v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rcx
  unsigned int v18; // ecx
  __int64 v19; // rdx
  __int64 v20; // rax
  int v21; // ecx
  __int64 v22; // r13
  __int64 v23; // rax
  unsigned int v24; // r8d
  __int64 v25; // rsi
  unsigned int v26; // r10d
  __int64 v27; // r9
  unsigned int v28; // edx
  __int64 *v29; // rax
  __int64 v30; // rdi
  unsigned int v31; // ecx
  __int64 v32; // rdx
  __int64 v33; // rax
  unsigned int v34; // esi
  __int64 v35; // rdi
  unsigned int v36; // edx
  __int64 *v37; // rax
  __int64 v38; // r9
  unsigned int v39; // ecx
  __int64 v40; // rdx
  __int64 v41; // rax
  int v42; // esi
  __int64 *v43; // rcx
  int v44; // eax
  __int64 v45; // r11
  int v46; // r11d
  __int64 v47; // r11
  int v48; // r11d
  int v49; // eax
  int v50; // r10d
  __int64 *v51; // rcx
  int v52; // eax
  int v53; // edx
  int v54; // r10d
  __int64 *v55; // r9
  int v56; // eax
  int v57; // edx
  __int64 v58; // rcx
  int v59; // ecx
  __int64 *v60; // r11
  __int64 *v61; // r11
  int v62; // [rsp+14h] [rbp-4Ch]
  unsigned int v63; // [rsp+18h] [rbp-48h]
  __int64 v64; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v65[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = a2;
  v6 = *(_BYTE *)(a2 + 16);
  if ( v6 != 78 )
  {
    if ( v6 != 77 )
      goto LABEL_9;
    v34 = *(_DWORD *)(a1 + 80);
    v64 = v5;
    v35 = *(_QWORD *)(a1 + 64);
    if ( v34 )
    {
      v36 = (v34 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v37 = (__int64 *)(v35 + 16LL * v36);
      v38 = *v37;
      if ( v5 == *v37 )
      {
        v39 = *((_DWORD *)v37 + 2);
LABEL_32:
        v40 = 1LL << v39;
        v41 = 8LL * (v39 >> 6);
        goto LABEL_33;
      }
      v50 = 1;
      v51 = 0;
      while ( v38 != -8 )
      {
        if ( v38 != -16 || v51 )
          v37 = v51;
        v36 = (v34 - 1) & (v50 + v36);
        v61 = (__int64 *)(v35 + 16LL * v36);
        v38 = *v61;
        if ( v5 == *v61 )
        {
          v39 = *((_DWORD *)v61 + 2);
          goto LABEL_32;
        }
        ++v50;
        v51 = v37;
        v37 = (__int64 *)(v35 + 16LL * v36);
      }
      if ( !v51 )
        v51 = v37;
      v52 = *(_DWORD *)(a1 + 72);
      ++*(_QWORD *)(a1 + 56);
      v53 = v52 + 1;
      if ( 4 * (v52 + 1) < 3 * v34 )
      {
        if ( v34 - *(_DWORD *)(a1 + 76) - v53 <= v34 >> 3 )
          goto LABEL_67;
LABEL_62:
        *(_DWORD *)(a1 + 72) = v53;
        if ( *v51 != -8 )
          --*(_DWORD *)(a1 + 76);
        *v51 = v5;
        v40 = 1;
        v41 = 0;
        *((_DWORD *)v51 + 2) = 0;
LABEL_33:
        result = *(_QWORD *)(a3 + 24) + v41;
        *(_QWORD *)result |= v40;
        return result;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 56);
    }
    v34 *= 2;
LABEL_67:
    sub_1BFE340(a1 + 56, v34);
    sub_1BFD9C0(a1 + 56, &v64, v65);
    v51 = (__int64 *)v65[0];
    v5 = v64;
    v53 = *(_DWORD *)(a1 + 72) + 1;
    goto LABEL_62;
  }
  v7 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v7 + 16)
    || (*(_BYTE *)(v7 + 33) & 0x20) == 0
    || (result = (unsigned int)(*(_DWORD *)(v7 + 36) - 35), (unsigned int)result > 3)
    && ((*(_BYTE *)(v7 + 33) & 0x20) == 0
     || (result = *(unsigned int *)(v7 + 36), (_DWORD)result != 4)
     && (result = (unsigned int)(result - 116), (unsigned int)result > 1)) )
  {
LABEL_9:
    v9 = *(_DWORD *)(a1 + 80);
    if ( !v9 )
    {
LABEL_14:
      v21 = *(_DWORD *)(v5 + 20);
      result = v21 & 0xFFFFFFF;
      if ( (v21 & 0xFFFFFFF) == 0 )
        return result;
      v22 = 0;
      while ( 1 )
      {
        if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
          v23 = *(_QWORD *)(v5 - 8);
        else
          v23 = v5 - 24LL * (unsigned int)result;
        v24 = *(_DWORD *)(a1 + 80);
        v25 = *(_QWORD *)(v23 + 24 * v22);
        v64 = v25;
        if ( !v24 )
          goto LABEL_25;
        v26 = v24 - 1;
        v27 = *(_QWORD *)(a1 + 64);
        v28 = (v24 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v29 = (__int64 *)(v27 + 16LL * v28);
        v30 = *v29;
        if ( v25 != *v29 )
        {
          v63 = (v24 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          v45 = *v29;
          v62 = 1;
          while ( v45 != -8 )
          {
            v46 = v62++;
            v47 = v26 & (v46 + v63);
            v63 = v47;
            v45 = *(_QWORD *)(v27 + 16 * v47);
            if ( v25 == v45 )
            {
              v29 = (__int64 *)(v27 + 16LL * (v26 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4))));
              goto LABEL_19;
            }
          }
          goto LABEL_25;
        }
LABEL_19:
        if ( *(_BYTE *)(v25 + 16) == 17 )
        {
          if ( sub_3953740(a1 + 176, v25) )
            goto LABEL_24;
          v24 = *(_DWORD *)(a1 + 80);
          if ( !v24 )
          {
            ++*(_QWORD *)(a1 + 56);
            goto LABEL_37;
          }
          v25 = v64;
          v26 = v24 - 1;
          v27 = *(_QWORD *)(a1 + 64);
          v28 = (v24 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
          v29 = (__int64 *)(v27 + 16LL * v28);
          v30 = *v29;
        }
        if ( v25 == v30 )
        {
          v31 = *((_DWORD *)v29 + 2);
LABEL_22:
          v32 = 1LL << v31;
          v33 = 8LL * (v31 >> 6);
          goto LABEL_23;
        }
        v48 = 1;
        v43 = 0;
        while ( v30 != -8 )
        {
          if ( v43 || v30 != -16 )
            v29 = v43;
          v59 = v48 + 1;
          v28 = v26 & (v48 + v28);
          v60 = (__int64 *)(v27 + 16LL * v28);
          v30 = *v60;
          if ( v25 == *v60 )
          {
            v31 = *((_DWORD *)v60 + 2);
            goto LABEL_22;
          }
          v48 = v59;
          v43 = v29;
          v29 = (__int64 *)(v27 + 16LL * v28);
        }
        if ( !v43 )
          v43 = v29;
        v49 = *(_DWORD *)(a1 + 72);
        ++*(_QWORD *)(a1 + 56);
        v44 = v49 + 1;
        if ( 4 * v44 < 3 * v24 )
        {
          if ( v24 - (v44 + *(_DWORD *)(a1 + 76)) > v24 >> 3 )
            goto LABEL_39;
          v42 = v24;
          goto LABEL_38;
        }
LABEL_37:
        v42 = 2 * v24;
LABEL_38:
        sub_1BFE340(a1 + 56, v42);
        sub_1BFD9C0(a1 + 56, &v64, v65);
        v43 = (__int64 *)v65[0];
        v25 = v64;
        v44 = *(_DWORD *)(a1 + 72) + 1;
LABEL_39:
        *(_DWORD *)(a1 + 72) = v44;
        if ( *v43 != -8 )
          --*(_DWORD *)(a1 + 76);
        *v43 = v25;
        v32 = 1;
        v33 = 0;
        *((_DWORD *)v43 + 2) = 0;
LABEL_23:
        *(_QWORD *)(*(_QWORD *)(a3 + 24) + v33) |= v32;
LABEL_24:
        v21 = *(_DWORD *)(v5 + 20);
LABEL_25:
        ++v22;
        result = v21 & 0xFFFFFFF;
        if ( (unsigned int)result <= (unsigned int)v22 )
          return result;
      }
    }
    v10 = v9 - 1;
    v11 = *(_QWORD *)(a1 + 64);
    v12 = 1;
    v13 = (v9 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v14 = v13;
    v15 = (__int64 *)(v11 + 16LL * v13);
    v16 = *v15;
    if ( v5 != *v15 )
    {
      while ( v16 != -8 )
      {
        v14 = v10 & (v12 + v14);
        v16 = *(_QWORD *)(v11 + 16LL * v14);
        if ( v5 == v16 )
          goto LABEL_11;
        ++v12;
      }
      goto LABEL_14;
    }
LABEL_11:
    v64 = v5;
    v17 = *v15;
    if ( v5 == *v15 )
    {
LABEL_12:
      v18 = *((_DWORD *)v15 + 2);
      v19 = ~(1LL << v18);
      v20 = 8LL * (v18 >> 6);
LABEL_13:
      *(_QWORD *)(*(_QWORD *)(a3 + 24) + v20) &= v19;
      goto LABEL_14;
    }
    v54 = 1;
    v55 = 0;
    while ( v17 != -8 )
    {
      if ( v17 == -16 && !v55 )
        v55 = v15;
      v13 = v10 & (v54 + v13);
      v15 = (__int64 *)(v11 + 16LL * v13);
      v17 = *v15;
      if ( v5 == *v15 )
        goto LABEL_12;
      ++v54;
    }
    if ( !v55 )
      v55 = v15;
    v56 = *(_DWORD *)(a1 + 72);
    ++*(_QWORD *)(a1 + 56);
    v57 = v56 + 1;
    if ( 4 * (v56 + 1) >= 3 * v9 )
    {
      v9 *= 2;
    }
    else
    {
      v58 = v5;
      if ( v9 - *(_DWORD *)(a1 + 76) - v57 > v9 >> 3 )
      {
LABEL_74:
        *(_DWORD *)(a1 + 72) = v57;
        if ( *v55 != -8 )
          --*(_DWORD *)(a1 + 76);
        *v55 = v58;
        v19 = -2;
        v20 = 0;
        *((_DWORD *)v55 + 2) = 0;
        goto LABEL_13;
      }
    }
    sub_1BFE340(a1 + 56, v9);
    sub_1BFD9C0(a1 + 56, &v64, v65);
    v55 = (__int64 *)v65[0];
    v58 = v64;
    v57 = *(_DWORD *)(a1 + 72) + 1;
    goto LABEL_74;
  }
  return result;
}
