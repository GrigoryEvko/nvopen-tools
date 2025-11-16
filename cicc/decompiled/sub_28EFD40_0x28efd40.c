// Function: sub_28EFD40
// Address: 0x28efd40
//
__int64 __fastcall sub_28EFD40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char *v5; // rbx
  int v6; // r15d
  char *v7; // rax
  unsigned int v8; // esi
  char *v9; // r13
  __int64 v10; // rdi
  int v11; // edx
  char *v12; // rax
  int *v13; // rax
  __int64 v14; // r9
  unsigned int v15; // edx
  __int64 v16; // rax
  char *v17; // r8
  int v18; // r13d
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 result; // rax
  unsigned int v22; // esi
  __int64 v23; // r14
  __int64 v24; // r8
  int v25; // r10d
  _QWORD *v26; // rdx
  unsigned int v27; // edi
  _QWORD *v28; // rax
  __int64 v29; // rcx
  int *v30; // rax
  __int64 v31; // r14
  __int64 v32; // rbx
  char *v33; // rdx
  char *v34; // r15
  __int64 v35; // r8
  unsigned int v36; // esi
  char *v37; // rdx
  __int64 v38; // rdi
  int v39; // ecx
  char *v40; // rax
  int *v41; // rax
  int v42; // eax
  int v43; // ecx
  __int64 v44; // r10
  unsigned int v45; // ecx
  __int64 v46; // rax
  char *v47; // r9
  int v48; // r15d
  int v49; // eax
  int v50; // eax
  int v51; // edi
  __int64 v52; // rsi
  unsigned int v53; // eax
  __int64 v54; // r8
  int v55; // r10d
  _QWORD *v56; // r9
  int v57; // ecx
  int v58; // eax
  int v59; // eax
  int v60; // eax
  __int64 v61; // rdi
  _QWORD *v62; // r8
  unsigned int v63; // ebx
  int v64; // r9d
  __int64 v65; // rsi
  __int64 v66; // rdx
  __int64 v67; // rcx
  int v68; // eax
  char *v69; // [rsp+0h] [rbp-90h]
  __int64 v70; // [rsp+8h] [rbp-88h]
  __int64 v71; // [rsp+10h] [rbp-80h]
  __int64 v72; // [rsp+10h] [rbp-80h]
  int v74; // [rsp+18h] [rbp-78h]
  __int64 i; // [rsp+20h] [rbp-70h]
  char *v76; // [rsp+28h] [rbp-68h]
  int v77; // [rsp+28h] [rbp-68h]
  __int64 v78; // [rsp+38h] [rbp-58h] BYREF
  __int64 v79; // [rsp+40h] [rbp-50h] BYREF
  __int64 v80; // [rsp+48h] [rbp-48h]
  char *v81; // [rsp+50h] [rbp-40h]

  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a2, a2, a3, a4);
    v5 = *(char **)(a2 + 96);
    v76 = &v5[40 * *(_QWORD *)(a2 + 104)];
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a2, a2, v66, v67);
      v5 = *(char **)(a2 + 96);
      if ( v5 != v76 )
        goto LABEL_4;
      goto LABEL_108;
    }
  }
  else
  {
    v5 = *(char **)(a2 + 96);
    v76 = &v5[40 * *(_QWORD *)(a2 + 104)];
  }
  if ( v5 != v76 )
  {
LABEL_4:
    v71 = a1 + 32;
    v6 = 2;
    while ( 1 )
    {
      v7 = v5;
      v79 = 0;
      ++v6;
      BYTE1(v7) = BYTE1(v5) & 0xEF;
      v80 = 0;
      v81 = v5;
      if ( v7 != (char *)-8192LL && v5 )
        sub_BD73F0((__int64)&v79);
      v8 = *(_DWORD *)(a1 + 56);
      if ( !v8 )
        break;
      v9 = v81;
      v14 = *(_QWORD *)(a1 + 40);
      v15 = (v8 - 1) & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
      v16 = v14 + 32LL * v15;
      v17 = *(char **)(v16 + 16);
      if ( v81 != v17 )
      {
        v57 = 1;
        v10 = 0;
        while ( v17 != (char *)-4096LL )
        {
          if ( v17 == (char *)-8192LL && !v10 )
            v10 = v16;
          v15 = (v8 - 1) & (v57 + v15);
          v16 = v14 + 32LL * v15;
          v17 = *(char **)(v16 + 16);
          if ( v81 == v17 )
            goto LABEL_22;
          ++v57;
        }
        if ( !v10 )
          v10 = v16;
        v58 = *(_DWORD *)(a1 + 48);
        ++*(_QWORD *)(a1 + 32);
        v11 = v58 + 1;
        v78 = v10;
        if ( 4 * (v58 + 1) < 3 * v8 )
        {
          if ( v8 - *(_DWORD *)(a1 + 52) - v11 <= v8 >> 3 )
          {
LABEL_11:
            sub_28EF240(v71, v8);
            sub_28EE370(v71, (__int64)&v79, &v78);
            v9 = v81;
            v10 = v78;
            v11 = *(_DWORD *)(a1 + 48) + 1;
          }
          *(_DWORD *)(a1 + 48) = v11;
          if ( *(_QWORD *)(v10 + 16) == -4096 )
          {
            if ( v9 != (char *)-4096LL )
              goto LABEL_17;
          }
          else
          {
            --*(_DWORD *)(a1 + 52);
            v12 = *(char **)(v10 + 16);
            if ( v12 != v9 )
            {
              if ( v12 + 4096 != 0 && v12 != 0 && v12 != (char *)-8192LL )
                sub_BD60C0((_QWORD *)v10);
LABEL_17:
              *(_QWORD *)(v10 + 16) = v9;
              if ( v9 + 4096 != 0 && v9 != 0 && v9 != (char *)-8192LL )
                sub_BD73F0(v10);
            }
          }
          *(_DWORD *)(v10 + 24) = 0;
          v13 = (int *)(v10 + 24);
          goto LABEL_23;
        }
LABEL_10:
        v8 *= 2;
        goto LABEL_11;
      }
LABEL_22:
      v13 = (int *)(v16 + 24);
LABEL_23:
      *v13 = v6;
      if ( v81 != 0 && v81 + 4096 != 0 && v81 != (char *)-8192LL )
        sub_BD60C0(&v79);
      v5 += 40;
      if ( v76 == v5 )
      {
        v18 = v6;
        goto LABEL_28;
      }
    }
    ++*(_QWORD *)(a1 + 32);
    v78 = 0;
    goto LABEL_10;
  }
LABEL_108:
  v18 = 2;
LABEL_28:
  v19 = *(_QWORD *)a3;
  v20 = *(unsigned int *)(a3 + 8);
  v74 = (v18 + 1) << 16;
  result = v19 + 8 * v20;
  v70 = v19;
  for ( i = result; v70 != i; v74 += 0x10000 )
  {
    v22 = *(_DWORD *)(a1 + 24);
    v23 = *(_QWORD *)(i - 8);
    if ( v22 )
    {
      v24 = *(_QWORD *)(a1 + 8);
      v25 = 1;
      v26 = 0;
      v27 = (v22 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v28 = (_QWORD *)(v24 + 16LL * v27);
      v29 = *v28;
      if ( v23 == *v28 )
      {
LABEL_31:
        v30 = (int *)(v28 + 1);
        goto LABEL_32;
      }
      while ( v29 != -4096 )
      {
        if ( !v26 && v29 == -8192 )
          v26 = v28;
        v27 = (v22 - 1) & (v25 + v27);
        v28 = (_QWORD *)(v24 + 16LL * v27);
        v29 = *v28;
        if ( v23 == *v28 )
          goto LABEL_31;
        ++v25;
      }
      if ( !v26 )
        v26 = v28;
      v42 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v43 = v42 + 1;
      if ( 4 * (v42 + 1) < 3 * v22 )
      {
        if ( v22 - *(_DWORD *)(a1 + 20) - v43 <= v22 >> 3 )
        {
          sub_B23080(a1, v22);
          v59 = *(_DWORD *)(a1 + 24);
          if ( !v59 )
          {
LABEL_129:
            ++*(_DWORD *)(a1 + 16);
            BUG();
          }
          v60 = v59 - 1;
          v61 = *(_QWORD *)(a1 + 8);
          v62 = 0;
          v63 = v60 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v64 = 1;
          v43 = *(_DWORD *)(a1 + 16) + 1;
          v26 = (_QWORD *)(v61 + 16LL * v63);
          v65 = *v26;
          if ( v23 != *v26 )
          {
            while ( v65 != -4096 )
            {
              if ( !v62 && v65 == -8192 )
                v62 = v26;
              v63 = v60 & (v64 + v63);
              v26 = (_QWORD *)(v61 + 16LL * v63);
              v65 = *v26;
              if ( v23 == *v26 )
                goto LABEL_69;
              ++v64;
            }
            if ( v62 )
              v26 = v62;
          }
        }
        goto LABEL_69;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_B23080(a1, 2 * v22);
    v50 = *(_DWORD *)(a1 + 24);
    if ( !v50 )
      goto LABEL_129;
    v51 = v50 - 1;
    v52 = *(_QWORD *)(a1 + 8);
    v53 = (v50 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
    v43 = *(_DWORD *)(a1 + 16) + 1;
    v26 = (_QWORD *)(v52 + 16LL * v53);
    v54 = *v26;
    if ( v23 != *v26 )
    {
      v55 = 1;
      v56 = 0;
      while ( v54 != -4096 )
      {
        if ( !v56 && v54 == -8192 )
          v56 = v26;
        v53 = v51 & (v55 + v53);
        v26 = (_QWORD *)(v52 + 16LL * v53);
        v54 = *v26;
        if ( v23 == *v26 )
          goto LABEL_69;
        ++v55;
      }
      if ( v56 )
        v26 = v56;
    }
LABEL_69:
    *(_DWORD *)(a1 + 16) = v43;
    if ( *v26 != -4096 )
      --*(_DWORD *)(a1 + 20);
    *v26 = v23;
    v30 = (int *)(v26 + 1);
    *((_DWORD *)v26 + 2) = 0;
LABEL_32:
    v31 = v23 + 48;
    *v30 = v74;
    v32 = *(_QWORD *)(v31 + 8);
    v77 = v74;
    if ( v31 != v32 )
    {
      while ( 1 )
      {
        v33 = (char *)(v32 - 24);
        if ( !v32 )
          v33 = 0;
        v34 = v33;
        if ( !(unsigned __int8)sub_991AB0(v33) )
          goto LABEL_34;
        ++v77;
        v35 = a1 + 32;
        v79 = 0;
        v80 = 0;
        v81 = v34;
        if ( v34 != 0 && v34 + 4096 != 0 && v34 != (char *)-8192LL )
        {
          sub_BD73F0((__int64)&v79);
          v35 = a1 + 32;
        }
        v36 = *(_DWORD *)(a1 + 56);
        if ( !v36 )
          break;
        v37 = v81;
        v44 = *(_QWORD *)(a1 + 40);
        v45 = (v36 - 1) & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
        v46 = v44 + 32LL * v45;
        v47 = *(char **)(v46 + 16);
        if ( v81 != v47 )
        {
          v48 = 1;
          v38 = 0;
          while ( v47 != (char *)-4096LL )
          {
            if ( v47 == (char *)-8192LL && !v38 )
              v38 = v46;
            v68 = v48++;
            v45 = (v36 - 1) & (v68 + v45);
            v46 = v44 + 32LL * v45;
            v47 = *(char **)(v46 + 16);
            if ( v81 == v47 )
              goto LABEL_73;
          }
          if ( !v38 )
            v38 = v46;
          v49 = *(_DWORD *)(a1 + 48);
          ++*(_QWORD *)(a1 + 32);
          v39 = v49 + 1;
          v78 = v38;
          if ( 4 * (v49 + 1) < 3 * v36 )
          {
            if ( v36 - *(_DWORD *)(a1 + 52) - v39 <= v36 >> 3 )
            {
LABEL_44:
              v72 = v35;
              sub_28EF240(v35, v36);
              sub_28EE370(v72, (__int64)&v79, &v78);
              v37 = v81;
              v38 = v78;
              v39 = *(_DWORD *)(a1 + 48) + 1;
            }
            *(_DWORD *)(a1 + 48) = v39;
            if ( *(_QWORD *)(v38 + 16) == -4096 )
            {
              if ( v37 != (char *)-4096LL )
                goto LABEL_50;
            }
            else
            {
              --*(_DWORD *)(a1 + 52);
              v40 = *(char **)(v38 + 16);
              if ( v40 != v37 )
              {
                if ( v40 + 4096 != 0 && v40 != 0 && v40 != (char *)-8192LL )
                {
                  v69 = v37;
                  sub_BD60C0((_QWORD *)v38);
                  v37 = v69;
                }
LABEL_50:
                *(_QWORD *)(v38 + 16) = v37;
                if ( v37 != 0 && v37 + 4096 != 0 && v37 != (char *)-8192LL )
                  sub_BD73F0(v38);
              }
            }
            *(_DWORD *)(v38 + 24) = 0;
            v41 = (int *)(v38 + 24);
            goto LABEL_54;
          }
LABEL_43:
          v36 *= 2;
          goto LABEL_44;
        }
LABEL_73:
        v41 = (int *)(v46 + 24);
LABEL_54:
        *v41 = v77;
        if ( v81 == 0 || v81 + 4096 == 0 || v81 == (char *)-8192LL )
        {
LABEL_34:
          v32 = *(_QWORD *)(v32 + 8);
          if ( v31 == v32 )
            goto LABEL_57;
        }
        else
        {
          sub_BD60C0(&v79);
          v32 = *(_QWORD *)(v32 + 8);
          if ( v31 == v32 )
            goto LABEL_57;
        }
      }
      ++*(_QWORD *)(a1 + 32);
      v78 = 0;
      goto LABEL_43;
    }
LABEL_57:
    i -= 8;
    result = i;
  }
  return result;
}
