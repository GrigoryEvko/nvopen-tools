// Function: sub_1075AF0
// Address: 0x1075af0
//
void __fastcall sub_1075AF0(__int64 a1, __int64 a2)
{
  __int64 *v2; // r14
  __int64 *v3; // rbx
  __int64 *v4; // rax
  __int64 v5; // rdx
  int v6; // r12d
  __int64 v8; // r13
  int v9; // eax
  unsigned int v10; // esi
  __int64 v11; // r10
  __int64 v12; // rcx
  __int64 v13; // r9
  __int64 *v14; // rax
  __int64 v15; // r8
  __int64 v16; // rsi
  __int64 *v17; // rbx
  int v18; // r14d
  __int64 v19; // r10
  __int64 v20; // rcx
  __int64 v21; // r9
  __int64 *v22; // rax
  __int64 v23; // r8
  __int64 v24; // r13
  unsigned int v25; // esi
  int v26; // esi
  int v27; // esi
  int v28; // eax
  __int64 *v29; // rdi
  __int64 *v30; // r10
  __int64 *v31; // rdi
  int v32; // eax
  int v33; // eax
  int v34; // eax
  int v35; // esi
  int v36; // esi
  __int64 *v37; // r10
  int v38; // esi
  int v39; // esi
  int v40; // esi
  int v41; // esi
  unsigned int v42; // edi
  unsigned int v43; // edi
  __int64 v44; // rax
  __int64 *v45; // rax
  __int64 v46; // rdx
  _QWORD *v47; // rax
  __int64 v48; // [rsp+8h] [rbp-A8h]
  __int64 v50; // [rsp+18h] [rbp-98h]
  __int64 *v51; // [rsp+18h] [rbp-98h]
  _QWORD v52[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v53; // [rsp+40h] [rbp-70h]
  _QWORD v54[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v55; // [rsp+70h] [rbp-40h]

  v2 = *(__int64 **)(a1 + 152);
  v3 = *(__int64 **)(a1 + 144);
  if ( v3 == v2 )
    return;
  v4 = *(__int64 **)(a1 + 144);
  do
  {
    v5 = *(unsigned __int8 *)(v4[1] + 164);
    if ( (unsigned int)(v5 - 6) > 2 && (_DWORD)v5 != 20 )
    {
      v44 = *v4;
      if ( (*(_BYTE *)(v44 + 8) & 1) != 0 )
      {
        v45 = *(__int64 **)(v44 - 8);
        v46 = *v45;
        v47 = v45 + 3;
      }
      else
      {
        v46 = 0;
        v47 = 0;
      }
      v52[2] = v47;
      v52[0] = "indirect symbol '";
      v54[0] = v52;
      v55 = 770;
      v53 = 1283;
      v52[3] = v46;
      v54[2] = "' not in a symbol pointer or stub section";
      sub_C64D30((__int64)v54, 1u);
    }
    v4 += 2;
  }
  while ( v2 != v4 );
  v50 = a1 + 168;
  v6 = 0;
  do
  {
    while ( 1 )
    {
      v8 = v3[1];
      v9 = *(unsigned __int8 *)(v8 + 164);
      if ( v9 == 20 || v9 == 6 )
        break;
      v3 += 2;
      ++v6;
      if ( v2 == v3 )
        goto LABEL_13;
    }
    v10 = *(_DWORD *)(a1 + 192);
    if ( v10 )
    {
      v11 = *(_QWORD *)(a1 + 176);
      v12 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
      v13 = (v10 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v14 = (__int64 *)(v11 + 16 * v13);
      v15 = *v14;
      if ( v8 == *v14 )
        goto LABEL_12;
      v5 = 1;
      v31 = 0;
      while ( v15 != -4096 )
      {
        if ( v31 || v15 != -8192 )
          v14 = v31;
        v42 = v5 + 1;
        v13 = (v10 - 1) & ((_DWORD)v5 + (_DWORD)v13);
        v5 = v11 + 16LL * (unsigned int)v13;
        v15 = *(_QWORD *)v5;
        if ( v8 == *(_QWORD *)v5 )
          goto LABEL_12;
        v5 = v42;
        v31 = v14;
        v14 = (__int64 *)(v11 + 16LL * (unsigned int)v13);
      }
      if ( !v31 )
        v31 = v14;
      v32 = *(_DWORD *)(a1 + 184);
      ++*(_QWORD *)(a1 + 168);
      v33 = v32 + 1;
      if ( 4 * v33 < 3 * v10 )
      {
        v15 = v10 - *(_DWORD *)(a1 + 188) - v33;
        v13 = v10 >> 3;
        if ( (unsigned int)v15 > (unsigned int)v13 )
          goto LABEL_36;
        sub_1075910(v50, v10);
        v40 = *(_DWORD *)(a1 + 192);
        if ( !v40 )
        {
LABEL_96:
          ++*(_DWORD *)(a1 + 184);
          BUG();
        }
        v41 = v40 - 1;
        v37 = 0;
        v13 = *(_QWORD *)(a1 + 176);
        v5 = 1;
        v12 = v41 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v33 = *(_DWORD *)(a1 + 184) + 1;
        v31 = (__int64 *)(v13 + 16 * v12);
        v15 = *v31;
        if ( v8 == *v31 )
          goto LABEL_36;
        while ( v15 != -4096 )
        {
          if ( v15 == -8192 && !v37 )
            v37 = v31;
          v12 = v41 & (unsigned int)(v5 + v12);
          v31 = (__int64 *)(v13 + 16LL * (unsigned int)v12);
          v15 = *v31;
          if ( v8 == *v31 )
            goto LABEL_36;
          v5 = (unsigned int)(v5 + 1);
        }
        goto LABEL_53;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 168);
    }
    sub_1075910(v50, 2 * v10);
    v35 = *(_DWORD *)(a1 + 192);
    if ( !v35 )
      goto LABEL_96;
    v36 = v35 - 1;
    v13 = *(_QWORD *)(a1 + 176);
    v12 = v36 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v33 = *(_DWORD *)(a1 + 184) + 1;
    v31 = (__int64 *)(v13 + 16 * v12);
    v15 = *v31;
    if ( v8 == *v31 )
      goto LABEL_36;
    v5 = 1;
    v37 = 0;
    while ( v15 != -4096 )
    {
      if ( !v37 && v15 == -8192 )
        v37 = v31;
      v12 = v36 & (unsigned int)(v5 + v12);
      v31 = (__int64 *)(v13 + 16LL * (unsigned int)v12);
      v15 = *v31;
      if ( v8 == *v31 )
        goto LABEL_36;
      v5 = (unsigned int)(v5 + 1);
    }
LABEL_53:
    if ( v37 )
      v31 = v37;
LABEL_36:
    *(_DWORD *)(a1 + 184) = v33;
    if ( *v31 != -4096 )
      --*(_DWORD *)(a1 + 188);
    *v31 = v8;
    *((_DWORD *)v31 + 2) = v6;
LABEL_12:
    v16 = *v3;
    v3 += 2;
    ++v6;
    sub_E5CB20(a2, v16, v5, v12, v15, v13);
  }
  while ( v2 != v3 );
LABEL_13:
  v17 = *(__int64 **)(a1 + 144);
  v51 = *(__int64 **)(a1 + 152);
  if ( v51 != v17 )
  {
    v18 = 0;
    v48 = a1 + 168;
    while ( 1 )
    {
      v24 = v17[1];
      if ( (unsigned int)*(unsigned __int8 *)(v24 + 164) - 7 <= 1 )
        break;
LABEL_18:
      ++v18;
      v17 += 2;
      if ( v51 == v17 )
        return;
    }
    v25 = *(_DWORD *)(a1 + 192);
    if ( !v25 )
    {
      ++*(_QWORD *)(a1 + 168);
      goto LABEL_22;
    }
    v19 = *(_QWORD *)(a1 + 176);
    v20 = ((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4);
    v21 = (v25 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
    v22 = (__int64 *)(v19 + 16 * v21);
    v23 = *v22;
    if ( v24 == *v22 )
    {
LABEL_16:
      if ( (unsigned __int8)sub_E5CB20(a2, *v17, v5, v20, v23, v21) )
        *(_WORD *)(*v17 + 12) |= 1u;
      goto LABEL_18;
    }
    v5 = 1;
    v29 = 0;
    while ( v23 != -4096 )
    {
      if ( v23 != -8192 || v29 )
        v22 = v29;
      v43 = v5 + 1;
      v21 = (v25 - 1) & ((_DWORD)v5 + (_DWORD)v21);
      v5 = v19 + 16LL * (unsigned int)v21;
      v23 = *(_QWORD *)v5;
      if ( v24 == *(_QWORD *)v5 )
        goto LABEL_16;
      v5 = v43;
      v29 = v22;
      v22 = (__int64 *)(v19 + 16LL * (unsigned int)v21);
    }
    if ( !v29 )
      v29 = v22;
    v34 = *(_DWORD *)(a1 + 184);
    ++*(_QWORD *)(a1 + 168);
    v28 = v34 + 1;
    if ( 4 * v28 >= 3 * v25 )
    {
LABEL_22:
      sub_1075910(v48, 2 * v25);
      v26 = *(_DWORD *)(a1 + 192);
      if ( !v26 )
        goto LABEL_96;
      v27 = v26 - 1;
      v21 = *(_QWORD *)(a1 + 176);
      v20 = v27 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v28 = *(_DWORD *)(a1 + 184) + 1;
      v29 = (__int64 *)(v21 + 16 * v20);
      v23 = *v29;
      if ( v24 != *v29 )
      {
        v5 = 1;
        v30 = 0;
        while ( v23 != -4096 )
        {
          if ( !v30 && v23 == -8192 )
            v30 = v29;
          v20 = v27 & (unsigned int)(v5 + v20);
          v29 = (__int64 *)(v21 + 16LL * (unsigned int)v20);
          v23 = *v29;
          if ( v24 == *v29 )
            goto LABEL_45;
          v5 = (unsigned int)(v5 + 1);
        }
        goto LABEL_26;
      }
    }
    else
    {
      v23 = v25 - *(_DWORD *)(a1 + 188) - v28;
      v21 = v25 >> 3;
      if ( (unsigned int)v23 <= (unsigned int)v21 )
      {
        sub_1075910(v48, v25);
        v38 = *(_DWORD *)(a1 + 192);
        if ( !v38 )
          goto LABEL_96;
        v39 = v38 - 1;
        v30 = 0;
        v21 = *(_QWORD *)(a1 + 176);
        v5 = 1;
        v20 = v39 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v28 = *(_DWORD *)(a1 + 184) + 1;
        v29 = (__int64 *)(v21 + 16 * v20);
        v23 = *v29;
        if ( v24 != *v29 )
        {
          while ( v23 != -4096 )
          {
            if ( v23 == -8192 && !v30 )
              v30 = v29;
            v20 = v39 & (unsigned int)(v5 + v20);
            v29 = (__int64 *)(v21 + 16LL * (unsigned int)v20);
            v23 = *v29;
            if ( v24 == *v29 )
              goto LABEL_45;
            v5 = (unsigned int)(v5 + 1);
          }
LABEL_26:
          if ( v30 )
            v29 = v30;
        }
      }
    }
LABEL_45:
    *(_DWORD *)(a1 + 184) = v28;
    if ( *v29 != -4096 )
      --*(_DWORD *)(a1 + 188);
    *v29 = v24;
    *((_DWORD *)v29 + 2) = v18;
    goto LABEL_16;
  }
}
