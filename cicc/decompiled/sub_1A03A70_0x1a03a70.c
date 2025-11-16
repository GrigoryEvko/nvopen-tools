// Function: sub_1A03A70
// Address: 0x1a03a70
//
__int64 __fastcall sub_1A03A70(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  unsigned __int8 v4; // al
  unsigned int v5; // esi
  __int64 v6; // rcx
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r9
  unsigned int v10; // r14d
  __int64 v12; // rdi
  __int64 *v13; // r8
  __int64 v14; // rcx
  int v15; // edx
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // r8
  __int64 v19; // rcx
  __int64 *v20; // rax
  __int64 v21; // r10
  unsigned int v22; // r15d
  __int64 v23; // r13
  __int64 v24; // rax
  unsigned int v25; // eax
  __int64 v26; // rdx
  __int64 v27; // rcx
  unsigned int v28; // esi
  __int64 v29; // rdi
  unsigned int v30; // edx
  __int64 *v31; // rax
  __int64 v32; // rcx
  unsigned int v33; // esi
  __int64 v34; // r13
  __int64 v35; // rcx
  unsigned int v36; // edx
  unsigned int *v37; // rax
  __int64 v38; // r9
  __int64 v39; // rdx
  __int64 v40; // rcx
  int v41; // r10d
  int v42; // eax
  int v43; // r11d
  __int64 *v44; // rdi
  int v45; // eax
  int v46; // r11d
  __int64 *v47; // r10
  int v48; // ecx
  int v49; // ecx
  int v50; // r10d
  unsigned int *v51; // r8
  int v52; // eax
  int v53; // edx
  __int64 *v54; // r13
  __int64 v55; // [rsp+0h] [rbp-50h]
  __int64 v56; // [rsp+8h] [rbp-48h]
  __int64 v57; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v58[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = a2;
  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 > 0x17u )
  {
    v57 = a2;
    v5 = *(_DWORD *)(a1 + 56);
    v55 = a1 + 32;
    if ( v5 )
    {
      v6 = *(_QWORD *)(a1 + 40);
      v7 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( v3 == *v8 )
      {
LABEL_4:
        v10 = *((_DWORD *)v8 + 2);
        if ( v10 )
          return v10;
LABEL_12:
        v16 = *(_QWORD *)(v3 + 40);
        v17 = *(unsigned int *)(a1 + 24);
        v57 = v16;
        if ( (_DWORD)v17 )
        {
          v18 = *(_QWORD *)(a1 + 8);
          v19 = ((_DWORD)v17 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v20 = (__int64 *)(v18 + 16 * v19);
          v21 = *v20;
          if ( v16 == *v20 )
          {
            v10 = *((_DWORD *)v20 + 2);
            goto LABEL_15;
          }
          v43 = 1;
          v44 = 0;
          while ( v21 != -8 )
          {
            if ( v21 != -16 || v44 )
              v20 = v44;
            v19 = ((_DWORD)v17 - 1) & (unsigned int)(v43 + v19);
            v54 = (__int64 *)(v18 + 16LL * (unsigned int)v19);
            v21 = *v54;
            if ( v16 == *v54 )
            {
              v10 = *((_DWORD *)v54 + 2);
              goto LABEL_15;
            }
            ++v43;
            v44 = v20;
            v20 = (__int64 *)(v18 + 16LL * (unsigned int)v19);
          }
          if ( !v44 )
            v44 = v20;
          v45 = *(_DWORD *)(a1 + 16);
          ++*(_QWORD *)a1;
          v19 = (unsigned int)(v45 + 1);
          if ( 4 * (int)v19 < (unsigned int)(3 * v17) )
          {
            if ( (int)v17 - *(_DWORD *)(a1 + 20) - (int)v19 <= (unsigned int)v17 >> 3 )
              goto LABEL_54;
LABEL_49:
            *(_DWORD *)(a1 + 16) = v19;
            if ( *v44 != -8 )
              --*(_DWORD *)(a1 + 20);
            *v44 = v16;
            v10 = 0;
            *((_DWORD *)v44 + 2) = 0;
LABEL_15:
            if ( (*(_DWORD *)(v3 + 20) & 0xFFFFFFF) != 0 )
            {
              v22 = 0;
              v23 = 0;
              v56 = 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
              while ( v22 != v10 )
              {
                if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
                  v24 = *(_QWORD *)(v3 - 8);
                else
                  v24 = v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
                v17 = *(_QWORD *)(v24 + v23);
                v25 = sub_1A03A70(a1, v17);
                if ( v22 < v25 )
                  v22 = v25;
                v23 += 24;
                if ( v23 == v56 )
                {
                  v10 = v22;
                  break;
                }
              }
            }
            else
            {
              v10 = 0;
            }
            if ( !sub_15FB730(v3, v17, v16, v19) && !sub_15FB6B0(v3, v17, v26, v27) )
              v10 += sub_15FB6D0(v3, 0, v39, v40) == 0;
            v28 = *(_DWORD *)(a1 + 56);
            v57 = v3;
            if ( v28 )
            {
              v29 = *(_QWORD *)(a1 + 40);
              v30 = (v28 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
              v31 = (__int64 *)(v29 + 16LL * v30);
              v32 = *v31;
              if ( v3 == *v31 )
              {
LABEL_27:
                *((_DWORD *)v31 + 2) = v10;
                return v10;
              }
              v46 = 1;
              v47 = 0;
              while ( v32 != -8 )
              {
                if ( v32 == -16 && !v47 )
                  v47 = v31;
                v30 = (v28 - 1) & (v46 + v30);
                v31 = (__int64 *)(v29 + 16LL * v30);
                v32 = *v31;
                if ( v3 == *v31 )
                  goto LABEL_27;
                ++v46;
              }
              v48 = *(_DWORD *)(a1 + 48);
              if ( v47 )
                v31 = v47;
              ++*(_QWORD *)(a1 + 32);
              v49 = v48 + 1;
              if ( 4 * v49 < 3 * v28 )
              {
                if ( v28 - *(_DWORD *)(a1 + 52) - v49 > v28 >> 3 )
                {
LABEL_61:
                  *(_DWORD *)(a1 + 48) = v49;
                  if ( *v31 != -8 )
                    --*(_DWORD *)(a1 + 52);
                  *v31 = v3;
                  *((_DWORD *)v31 + 2) = 0;
                  goto LABEL_27;
                }
LABEL_66:
                sub_1A038B0(v55, v28);
                sub_1A02780(v55, &v57, v58);
                v31 = (__int64 *)v58[0];
                v3 = v57;
                v49 = *(_DWORD *)(a1 + 48) + 1;
                goto LABEL_61;
              }
            }
            else
            {
              ++*(_QWORD *)(a1 + 32);
            }
            v28 *= 2;
            goto LABEL_66;
          }
        }
        else
        {
          ++*(_QWORD *)a1;
        }
        LODWORD(v17) = 2 * v17;
LABEL_54:
        sub_13FEAC0(a1, v17);
        v17 = (__int64)&v57;
        sub_13FDDE0(a1, &v57, v58);
        v44 = (__int64 *)v58[0];
        v16 = v57;
        v19 = (unsigned int)(*(_DWORD *)(a1 + 16) + 1);
        goto LABEL_49;
      }
      v41 = 1;
      v13 = 0;
      while ( v9 != -8 )
      {
        if ( !v13 && v9 == -16 )
          v13 = v8;
        v7 = (v5 - 1) & (v41 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( v3 == *v8 )
          goto LABEL_4;
        ++v41;
      }
      if ( !v13 )
        v13 = v8;
      v42 = *(_DWORD *)(a1 + 48);
      ++*(_QWORD *)(a1 + 32);
      v15 = v42 + 1;
      if ( 4 * (v42 + 1) < 3 * v5 )
      {
        v14 = v3;
        if ( v5 - *(_DWORD *)(a1 + 52) - v15 > v5 >> 3 )
          goto LABEL_9;
        sub_1A038B0(v55, v5);
        v12 = a1 + 32;
LABEL_8:
        sub_1A02780(v12, &v57, v58);
        v13 = (__int64 *)v58[0];
        v14 = v57;
        v15 = *(_DWORD *)(a1 + 48) + 1;
LABEL_9:
        *(_DWORD *)(a1 + 48) = v15;
        if ( *v13 != -8 )
          --*(_DWORD *)(a1 + 52);
        *v13 = v14;
        *((_DWORD *)v13 + 2) = 0;
        goto LABEL_12;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 32);
    }
    sub_1A038B0(v55, 2 * v5);
    v12 = a1 + 32;
    goto LABEL_8;
  }
  v10 = 0;
  if ( v4 != 17 )
    return v10;
  v57 = a2;
  v33 = *(_DWORD *)(a1 + 56);
  v34 = a1 + 32;
  if ( !v33 )
  {
    ++*(_QWORD *)(a1 + 32);
LABEL_77:
    v33 *= 2;
    goto LABEL_78;
  }
  v35 = *(_QWORD *)(a1 + 40);
  v36 = (v33 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v37 = (unsigned int *)(v35 + 16LL * v36);
  v38 = *(_QWORD *)v37;
  if ( v3 == *(_QWORD *)v37 )
    return v37[2];
  v50 = 1;
  v51 = 0;
  while ( v38 != -8 )
  {
    if ( !v51 && v38 == -16 )
      v51 = v37;
    v36 = (v33 - 1) & (v50 + v36);
    v37 = (unsigned int *)(v35 + 16LL * v36);
    v38 = *(_QWORD *)v37;
    if ( v3 == *(_QWORD *)v37 )
      return v37[2];
    ++v50;
  }
  if ( !v51 )
    v51 = v37;
  v52 = *(_DWORD *)(a1 + 48);
  ++*(_QWORD *)(a1 + 32);
  v53 = v52 + 1;
  if ( 4 * (v52 + 1) >= 3 * v33 )
    goto LABEL_77;
  if ( v33 - *(_DWORD *)(a1 + 52) - v53 <= v33 >> 3 )
  {
LABEL_78:
    sub_1A038B0(v34, v33);
    sub_1A02780(v34, &v57, v58);
    v51 = (unsigned int *)v58[0];
    v3 = v57;
    v53 = *(_DWORD *)(a1 + 48) + 1;
  }
  *(_DWORD *)(a1 + 48) = v53;
  if ( *(_QWORD *)v51 != -8 )
    --*(_DWORD *)(a1 + 52);
  *(_QWORD *)v51 = v3;
  v10 = 0;
  v51[2] = 0;
  return v10;
}
