// Function: sub_F75080
// Address: 0xf75080
//
_QWORD *__fastcall sub_F75080(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v5; // r15
  __int64 v9; // rax
  _QWORD *v10; // r13
  _BYTE *v11; // rsi
  __int64 *v12; // rbx
  __int64 v13; // rax
  __int64 *v15; // r13
  __int64 v16; // r15
  int v17; // esi
  __int64 v18; // rdi
  __int64 v19; // r8
  int v20; // esi
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // r10
  unsigned int v24; // esi
  int v25; // edx
  __int64 v26; // rax
  _QWORD *v27; // r8
  int v28; // edi
  __int64 v29; // rdx
  unsigned __int64 *v30; // rdi
  __int64 v31; // rdx
  _QWORD *v32; // rdx
  __int64 v33; // rax
  __int64 *v34; // rbx
  __int64 *i; // r15
  __int64 v36; // rdi
  __int64 v38; // r10
  unsigned int v39; // edi
  _QWORD *v40; // rdx
  __int64 v41; // r9
  int v42; // eax
  int v43; // ecx
  _BYTE *v44; // rsi
  int v45; // ecx
  int v46; // ecx
  int v47; // esi
  int v48; // esi
  __int64 v49; // r9
  int v50; // ecx
  _QWORD *v51; // r10
  unsigned int v52; // edx
  __int64 v53; // rdi
  int v54; // edx
  __int64 v55; // r9
  int v56; // ecx
  unsigned int v57; // esi
  __int64 v58; // rdi
  _QWORD *v59; // [rsp+8h] [rbp-88h]
  unsigned __int64 *v60; // [rsp+10h] [rbp-80h]
  _QWORD *v61; // [rsp+10h] [rbp-80h]
  _QWORD *v62; // [rsp+10h] [rbp-80h]
  __int64 *v64; // [rsp+28h] [rbp-68h]
  void *v65; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v66[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v67; // [rsp+48h] [rbp-48h]
  __int64 v68; // [rsp+50h] [rbp-40h]

  v5 = a1;
  v9 = *(_QWORD *)(a4 + 56);
  *(_QWORD *)(a4 + 136) += 160LL;
  v10 = (_QWORD *)((v9 + 7) & 0xFFFFFFFFFFFFFFF8LL);
  if ( *(_QWORD *)(a4 + 64) >= (unsigned __int64)(v10 + 20) && v9 )
    *(_QWORD *)(a4 + 56) = v10 + 20;
  else
    v10 = (_QWORD *)sub_9D1E70(a4 + 56, 160, 160, 3);
  memset(v10, 0, 0xA0u);
  v10[9] = 8;
  v10[8] = v10 + 11;
  *((_BYTE *)v10 + 84) = 1;
  v65 = v10;
  if ( a2 )
  {
    *v10 = a2;
    v11 = *(_BYTE **)(a2 + 16);
    if ( v11 == *(_BYTE **)(a2 + 24) )
    {
      sub_D4C7F0(a2 + 8, v11, &v65);
    }
    else
    {
      if ( v11 )
      {
        *(_QWORD *)v11 = v65;
        v11 = *(_BYTE **)(a2 + 16);
      }
      *(_QWORD *)(a2 + 16) = v11 + 8;
    }
  }
  else
  {
    v44 = *(_BYTE **)(a4 + 40);
    if ( v44 == *(_BYTE **)(a4 + 48) )
    {
      sub_D4C7F0(a4 + 32, v44, &v65);
    }
    else
    {
      if ( v44 )
      {
        *(_QWORD *)v44 = v10;
        v44 = *(_BYTE **)(a4 + 40);
      }
      *(_QWORD *)(a4 + 40) = v44 + 8;
    }
  }
  if ( a5 )
    sub_D5B000(a5, v10);
  v12 = *(__int64 **)(a1 + 32);
  if ( v12 != *(__int64 **)(a1 + 40) )
  {
    v13 = a3;
    v64 = v10;
    v15 = *(__int64 **)(a1 + 40);
    v16 = v13;
    while ( 1 )
    {
      v17 = *(_DWORD *)(a4 + 24);
      v18 = *v12;
      v19 = *(_QWORD *)(a4 + 8);
      if ( v17 )
      {
        v20 = v17 - 1;
        v21 = v20 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v22 = (__int64 *)(v19 + 16LL * v21);
        v23 = *v22;
        if ( v18 != *v22 )
        {
          v42 = 1;
          while ( v23 != -4096 )
          {
            v43 = v42 + 1;
            v21 = v20 & (v42 + v21);
            v22 = (__int64 *)(v19 + 16LL * v21);
            v23 = *v22;
            if ( v18 == *v22 )
              goto LABEL_16;
            v42 = v43;
          }
          goto LABEL_13;
        }
LABEL_16:
        if ( a1 == v22[1] )
          break;
      }
LABEL_13:
      if ( v15 == ++v12 )
      {
        v33 = v16;
        v10 = v64;
        v5 = a1;
        a3 = v33;
        goto LABEL_35;
      }
    }
    v66[0] = 2;
    v66[1] = 0;
    v67 = v18;
    if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
      sub_BD73F0((__int64)v66);
    v24 = *(_DWORD *)(v16 + 24);
    v68 = v16;
    v65 = &unk_49DD7B0;
    if ( v24 )
    {
      v26 = v67;
      v38 = *(_QWORD *)(v16 + 8);
      v39 = (v24 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
      v40 = (_QWORD *)(v38 + ((unsigned __int64)v39 << 6));
      v41 = v40[3];
      if ( v67 == v41 )
      {
LABEL_39:
        v32 = v40 + 5;
LABEL_40:
        v65 = &unk_49DB368;
        if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
        {
          v62 = v32;
          sub_BD60C0(v66);
          v32 = v62;
        }
        sub_D4F330(v64, v32[2], a4);
        goto LABEL_13;
      }
      v45 = 1;
      v27 = 0;
      while ( v41 != -4096 )
      {
        if ( !v27 && v41 == -8192 )
          v27 = v40;
        v39 = (v24 - 1) & (v45 + v39);
        v40 = (_QWORD *)(v38 + ((unsigned __int64)v39 << 6));
        v41 = v40[3];
        if ( v67 == v41 )
          goto LABEL_39;
        ++v45;
      }
      v46 = *(_DWORD *)(v16 + 16);
      if ( !v27 )
        v27 = v40;
      ++*(_QWORD *)v16;
      v28 = v46 + 1;
      if ( 4 * (v46 + 1) < 3 * v24 )
      {
        if ( v24 - *(_DWORD *)(v16 + 20) - v28 > v24 >> 3 )
        {
LABEL_25:
          *(_DWORD *)(v16 + 16) = v28;
          if ( v27[3] == -4096 )
          {
            v30 = v27 + 1;
            if ( v26 != -4096 )
            {
LABEL_30:
              v27[3] = v26;
              if ( v26 == 0 || v26 == -4096 || v26 == -8192 )
              {
                v26 = v67;
              }
              else
              {
                v61 = v27;
                sub_BD6050(v30, v66[0] & 0xFFFFFFFFFFFFFFF8LL);
                v26 = v67;
                v27 = v61;
              }
            }
          }
          else
          {
            --*(_DWORD *)(v16 + 20);
            v29 = v27[3];
            if ( v26 != v29 )
            {
              v30 = v27 + 1;
              if ( v29 != 0 && v29 != -4096 && v29 != -8192 )
              {
                v59 = v27;
                v60 = v27 + 1;
                sub_BD60C0(v30);
                v26 = v67;
                v27 = v59;
                v30 = v60;
              }
              goto LABEL_30;
            }
          }
          v31 = v68;
          v27[5] = 6;
          v27[6] = 0;
          v27[4] = v31;
          v32 = v27 + 5;
          v27[7] = 0;
          goto LABEL_40;
        }
        sub_CF32C0(v16, v24);
        v47 = *(_DWORD *)(v16 + 24);
        if ( !v47 )
          goto LABEL_23;
        v26 = v67;
        v48 = v47 - 1;
        v49 = *(_QWORD *)(v16 + 8);
        v50 = 1;
        v51 = 0;
        v52 = v48 & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
        v27 = (_QWORD *)(v49 + ((unsigned __int64)v52 << 6));
        v53 = v27[3];
        if ( v67 == v53 )
          goto LABEL_24;
        while ( v53 != -4096 )
        {
          if ( !v51 && v53 == -8192 )
            v51 = v27;
          v52 = v48 & (v50 + v52);
          v27 = (_QWORD *)(v49 + ((unsigned __int64)v52 << 6));
          v53 = v27[3];
          if ( v67 == v53 )
            goto LABEL_24;
          ++v50;
        }
        goto LABEL_74;
      }
    }
    else
    {
      ++*(_QWORD *)v16;
    }
    sub_CF32C0(v16, 2 * v24);
    v25 = *(_DWORD *)(v16 + 24);
    if ( !v25 )
    {
LABEL_23:
      v26 = v67;
      v27 = 0;
LABEL_24:
      v28 = *(_DWORD *)(v16 + 16) + 1;
      goto LABEL_25;
    }
    v26 = v67;
    v54 = v25 - 1;
    v55 = *(_QWORD *)(v16 + 8);
    v56 = 1;
    v51 = 0;
    v57 = v54 & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
    v27 = (_QWORD *)(v55 + ((unsigned __int64)v57 << 6));
    v58 = v27[3];
    if ( v67 == v58 )
      goto LABEL_24;
    while ( v58 != -4096 )
    {
      if ( !v51 && v58 == -8192 )
        v51 = v27;
      v57 = v54 & (v56 + v57);
      v27 = (_QWORD *)(v55 + ((unsigned __int64)v57 << 6));
      v58 = v27[3];
      if ( v67 == v58 )
        goto LABEL_24;
      ++v56;
    }
LABEL_74:
    if ( v51 )
      v27 = v51;
    goto LABEL_24;
  }
LABEL_35:
  v34 = *(__int64 **)(v5 + 16);
  for ( i = *(__int64 **)(v5 + 8); v34 != i; ++i )
  {
    v36 = *i;
    sub_F75080(v36, v10, a3, a4, a5);
  }
  return v10;
}
