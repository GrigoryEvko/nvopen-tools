// Function: sub_1BAAE10
// Address: 0x1baae10
//
__int64 __fastcall sub_1BAAE10(__int64 a1, __int64 a2, int *a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v8; // r12
  __int64 v9; // r14
  __int64 v10; // rax
  bool v11; // dl
  __int64 v12; // rdx
  __int64 *v13; // r12
  __int64 v14; // r11
  __int64 v15; // rax
  __int64 v16; // r11
  __int64 *v17; // r14
  __int64 v18; // rax
  __int64 v19; // rsi
  int v20; // r9d
  __int64 v21; // rdi
  unsigned int v22; // ecx
  __int64 *v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rax
  char v26; // r8
  __int64 *v27; // rax
  int v28; // r8d
  int v29; // r9d
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // r8d
  int v34; // r9d
  __int64 v35; // r12
  __int64 *v36; // rsi
  __int64 *v37; // rdx
  __int64 *v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rax
  int v42; // edi
  unsigned int v43; // esi
  int v44; // edx
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r10
  __int64 *v48; // rdx
  __int64 *v49; // r10
  int v50; // ecx
  int v51; // ecx
  int i; // [rsp+8h] [rbp-A8h]
  int v53; // [rsp+8h] [rbp-A8h]
  unsigned int v54; // [rsp+10h] [rbp-A0h]
  __int64 v55; // [rsp+10h] [rbp-A0h]
  __int64 *v56; // [rsp+10h] [rbp-A0h]
  char v59; // [rsp+37h] [rbp-79h]
  __int64 v60; // [rsp+38h] [rbp-78h] BYREF
  __int64 v61[2]; // [rsp+40h] [rbp-70h] BYREF
  __int16 v62; // [rsp+50h] [rbp-60h]
  __int64 *v63; // [rsp+60h] [rbp-50h] BYREF
  __int64 *v64; // [rsp+68h] [rbp-48h]
  _QWORD v65[8]; // [rsp+70h] [rbp-40h] BYREF

  v64 = &v60;
  v60 = a2;
  v63 = (__int64 *)a1;
  v65[1] = sub_1B961A0;
  v65[0] = sub_1B8E280;
  v8 = sub_1B932A0((__int64)&v63, a3, (__int64)a3);
  if ( v65[0] )
    ((void (__fastcall *)(__int64 **, __int64 **, __int64))v65[0])(&v63, &v63, 3);
  v59 = sub_1B91FD0(*(_QWORD *)(a1 + 32), v60);
  v9 = sub_22077B0(56);
  v10 = v60;
  if ( v9 )
  {
    *(_QWORD *)(v9 + 8) = 0;
    *(_QWORD *)(v9 + 16) = 0;
    *(_BYTE *)(v9 + 24) = 5;
    *(_QWORD *)v9 = &unk_49F6FA8;
    v11 = 0;
    *(_QWORD *)(v9 + 32) = 0;
    *(_QWORD *)(v9 + 40) = v10;
    *(_BYTE *)(v9 + 48) = v8;
    *(_BYTE *)(v9 + 49) = v59;
    if ( v59 )
      v11 = *(_QWORD *)(v10 + 8) != 0;
    *(_BYTE *)(v9 + 50) = v11;
  }
  v12 = 3LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
  {
    v13 = *(__int64 **)(v10 - 8);
    v14 = (__int64)&v13[v12];
  }
  else
  {
    v14 = v10;
    v13 = (__int64 *)(v10 - v12 * 8);
  }
  if ( v13 != (__int64 *)v14 )
  {
    v15 = v14;
    v16 = v9;
    v17 = (__int64 *)v15;
    while ( 1 )
    {
      v18 = *v13;
      if ( *(_BYTE *)(*v13 + 16) > 0x17u )
      {
        v19 = *(unsigned int *)(a5 + 24);
        v61[0] = *v13;
        if ( (_DWORD)v19 )
        {
          v20 = v19 - 1;
          v21 = *(_QWORD *)(a5 + 8);
          v22 = (v19 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v23 = (__int64 *)(v21 + 16LL * v22);
          v24 = *v23;
          if ( v18 != *v23 )
          {
            v54 = (v19 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
            v47 = *v23;
            for ( i = 1; ; ++i )
            {
              if ( v47 == -8 )
                goto LABEL_17;
              v54 = v20 & (i + v54);
              v47 = *(_QWORD *)(v21 + 16LL * v54);
              if ( v18 == v47 )
                break;
            }
            v48 = (__int64 *)(v21 + 16LL * (v20 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4))));
            if ( v21 + 16LL * v54 == v21 + 16LL * (unsigned int)v19 )
              goto LABEL_17;
            v53 = 1;
            v49 = 0;
            while ( v24 != -8 )
            {
              if ( v49 || v24 != -16 )
                v48 = v49;
              v22 = v20 & (v53 + v22);
              v56 = (__int64 *)(v21 + 16LL * v22);
              v24 = *v56;
              if ( v18 == *v56 )
              {
                v25 = v56[1];
                goto LABEL_16;
              }
              ++v53;
              v49 = v48;
              v48 = (__int64 *)(v21 + 16LL * v22);
            }
            v50 = *(_DWORD *)(a5 + 16);
            if ( !v49 )
              v49 = v48;
            ++*(_QWORD *)a5;
            v51 = v50 + 1;
            if ( 4 * v51 >= (unsigned int)(3 * v19) )
            {
              v55 = v16;
              LODWORD(v19) = 2 * v19;
            }
            else
            {
              if ( (int)v19 - *(_DWORD *)(a5 + 20) - v51 > (unsigned int)v19 >> 3 )
              {
LABEL_52:
                *(_DWORD *)(a5 + 16) = v51;
                if ( *v49 != -8 )
                  --*(_DWORD *)(a5 + 20);
                *v49 = v18;
                v25 = 0;
                v49[1] = 0;
                goto LABEL_16;
              }
              v55 = v16;
            }
            sub_1BAAC80(a5, v19);
            sub_1BA1090(a5, v61, &v63);
            v49 = v63;
            v18 = v61[0];
            v16 = v55;
            v51 = *(_DWORD *)(a5 + 16) + 1;
            goto LABEL_52;
          }
          if ( v23 != (__int64 *)(v21 + 16 * v19) )
          {
            v25 = v23[1];
LABEL_16:
            *(_BYTE *)(v25 + 50) = 0;
          }
        }
      }
LABEL_17:
      v13 += 3;
      if ( v17 == v13 )
      {
        v9 = v16;
        break;
      }
    }
  }
  if ( !v59 )
  {
    v46 = *(_QWORD *)(a4 + 112);
    *(_QWORD *)(v9 + 32) = a4;
    *(_QWORD *)(v9 + 16) = a4 + 112;
    v35 = a4;
    v46 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v9 + 8) = v46 | *(_QWORD *)(v9 + 8) & 7LL;
    *(_QWORD *)(v46 + 8) = v9 + 8;
    *(_QWORD *)(a4 + 112) = *(_QWORD *)(a4 + 112) & 7LL | (v9 + 8);
    return v35;
  }
  v26 = sub_1BA1090(a5, &v60, &v63);
  v27 = v63;
  if ( !v26 )
  {
    v42 = *(_DWORD *)(a5 + 16);
    v43 = *(_DWORD *)(a5 + 24);
    ++*(_QWORD *)a5;
    v44 = v42 + 1;
    if ( 4 * (v42 + 1) >= 3 * v43 )
    {
      v43 *= 2;
    }
    else if ( v43 - *(_DWORD *)(a5 + 20) - v44 > v43 >> 3 )
    {
LABEL_38:
      *(_DWORD *)(a5 + 16) = v44;
      if ( *v27 != -8 )
        --*(_DWORD *)(a5 + 20);
      v45 = v60;
      v27[1] = 0;
      *v27 = v45;
      goto LABEL_21;
    }
    sub_1BAAC80(a5, v43);
    sub_1BA1090(a5, &v60, &v63);
    v27 = v63;
    v44 = *(_DWORD *)(a5 + 16) + 1;
    goto LABEL_38;
  }
LABEL_21:
  v27[1] = v9;
  v30 = sub_1BAA750(a1, v60, (_QWORD *)v9, a6);
  v31 = *(unsigned int *)(a4 + 88);
  if ( (unsigned int)v31 >= *(_DWORD *)(a4 + 92) )
  {
    sub_16CD150(a4 + 80, (const void *)(a4 + 96), 0, 8, v28, v29);
    v31 = *(unsigned int *)(a4 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(a4 + 80) + 8 * v31) = v30;
  ++*(_DWORD *)(a4 + 88);
  v32 = *(unsigned int *)(v30 + 64);
  if ( (unsigned int)v32 >= *(_DWORD *)(v30 + 68) )
  {
    sub_16CD150(v30 + 56, (const void *)(v30 + 72), 0, 8, v28, v29);
    v32 = *(unsigned int *)(v30 + 64);
  }
  *(_QWORD *)(*(_QWORD *)(v30 + 56) + 8 * v32) = a4;
  ++*(_DWORD *)(v30 + 64);
  *(_QWORD *)(v30 + 48) = *(_QWORD *)(a4 + 48);
  v62 = 257;
  v35 = sub_22077B0(128);
  if ( v35 )
  {
    sub_16E2FC0((__int64 *)&v63, (__int64)v61);
    v36 = v63;
    *(_BYTE *)(v35 + 8) = 0;
    v37 = v64;
    *(_QWORD *)v35 = &unk_49F6D50;
    *(_QWORD *)(v35 + 16) = v35 + 32;
    sub_1B8E960((__int64 *)(v35 + 16), v36, (__int64)v37 + (_QWORD)v36);
    v38 = v63;
    *(_QWORD *)(v35 + 56) = v35 + 72;
    *(_QWORD *)(v35 + 64) = 0x100000000LL;
    *(_QWORD *)(v35 + 88) = 0x100000000LL;
    *(_QWORD *)(v35 + 48) = 0;
    *(_QWORD *)(v35 + 80) = v35 + 96;
    *(_QWORD *)(v35 + 104) = 0;
    if ( v38 != v65 )
      j_j___libc_free_0(v38, v65[0] + 1LL);
    *(_QWORD *)v35 = &unk_49F7110;
    *(_QWORD *)(v35 + 120) = v35 + 112;
    *(_QWORD *)(v35 + 112) = (v35 + 112) | 4;
  }
  v39 = *(unsigned int *)(v30 + 88);
  if ( (unsigned int)v39 >= *(_DWORD *)(v30 + 92) )
  {
    sub_16CD150(v30 + 80, (const void *)(v30 + 96), 0, 8, v33, v34);
    v39 = *(unsigned int *)(v30 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(v30 + 80) + 8 * v39) = v35;
  ++*(_DWORD *)(v30 + 88);
  v40 = *(unsigned int *)(v35 + 64);
  if ( (unsigned int)v40 >= *(_DWORD *)(v35 + 68) )
  {
    sub_16CD150(v35 + 56, (const void *)(v35 + 72), 0, 8, v33, v34);
    v40 = *(unsigned int *)(v35 + 64);
  }
  *(_QWORD *)(*(_QWORD *)(v35 + 56) + 8 * v40) = v30;
  ++*(_DWORD *)(v35 + 64);
  *(_QWORD *)(v35 + 48) = *(_QWORD *)(v30 + 48);
  return v35;
}
