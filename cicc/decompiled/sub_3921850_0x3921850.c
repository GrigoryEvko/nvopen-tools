// Function: sub_3921850
// Address: 0x3921850
//
__int64 __fastcall sub_3921850(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 result; // rax
  __int64 v4; // r12
  size_t v6; // r13
  unsigned __int64 v7; // r15
  _QWORD *v8; // r14
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // r14
  char v12; // si
  char v13; // al
  char *v14; // rax
  _QWORD *v15; // r9
  void *v16; // rdi
  __int64 v17; // rax
  _QWORD *v18; // r13
  __int64 v19; // rax
  unsigned int v20; // esi
  __int64 v21; // rdi
  __int64 v22; // r10
  unsigned int v23; // edx
  _QWORD *v24; // rax
  __int64 v25; // r9
  __int64 v26; // rsi
  __int64 v27; // rcx
  int v28; // r14d
  _QWORD *v29; // rcx
  int v30; // eax
  int v31; // eax
  __int64 v32; // rax
  int v33; // esi
  __int64 v34; // rdi
  int v35; // esi
  __int64 v36; // r10
  __int64 v37; // rdx
  __int64 v38; // r9
  int v39; // r14d
  _QWORD *v40; // r13
  int v41; // esi
  __int64 v42; // rdi
  int v43; // esi
  __int64 v44; // r10
  int v45; // r14d
  __int64 v46; // rdx
  __int64 v47; // r9
  __int64 *v48; // r13
  __int64 v49; // [rsp+0h] [rbp-90h]
  __int64 v50; // [rsp+8h] [rbp-88h]
  __int64 v53; // [rsp+20h] [rbp-70h]
  char *src; // [rsp+28h] [rbp-68h]
  _QWORD *srca; // [rsp+28h] [rbp-68h]
  __int64 v56; // [rsp+30h] [rbp-60h]
  _QWORD v57[2]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v58; // [rsp+50h] [rbp-40h]
  int v59; // [rsp+58h] [rbp-38h]

  result = *(_QWORD *)(a1 + 232);
  v4 = *(_QWORD *)(a1 + 224);
  v49 = a1 + 248;
  v53 = result;
  while ( v53 != v4 )
  {
    v6 = *(_QWORD *)(v4 + 8);
    v56 = *(_QWORD *)(v4 + 16);
    v7 = v6;
    src = *(char **)v4;
    sub_391B370(a1, (__int64)v57, 0);
    v8 = *(_QWORD **)(a1 + 8);
    v9 = (*(__int64 (__fastcall **)(_QWORD *))(*v8 + 64LL))(v8);
    v10 = v8[3] - v8[1];
    v11 = *(_QWORD *)(a1 + 8);
    v57[1] = v9 + v10;
    do
    {
      while ( 1 )
      {
        v12 = v7 & 0x7F;
        v13 = v7 & 0x7F | 0x80;
        v7 >>= 7;
        if ( v7 )
          v12 = v13;
        v14 = *(char **)(v11 + 24);
        if ( (unsigned __int64)v14 >= *(_QWORD *)(v11 + 16) )
          break;
        *(_QWORD *)(v11 + 24) = v14 + 1;
        *v14 = v12;
        if ( !v7 )
          goto LABEL_8;
      }
      sub_16E7DE0(v11, v12);
    }
    while ( v7 );
LABEL_8:
    v15 = *(_QWORD **)(a1 + 8);
    v16 = (void *)v15[3];
    if ( v6 > v15[2] - (_QWORD)v16 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 8), src, v6);
      v15 = *(_QWORD **)(a1 + 8);
    }
    else if ( v6 )
    {
      v50 = *(_QWORD *)(a1 + 8);
      memcpy(v16, src, v6);
      *(_QWORD *)(v50 + 24) += v6;
      v15 = *(_QWORD **)(a1 + 8);
    }
    srca = v15;
    v17 = (*(__int64 (__fastcall **)(_QWORD *))(*v15 + 64LL))(v15);
    v18 = *(_QWORD **)(a1 + 8);
    v58 = v17 + srca[3] - srca[1];
    v19 = (*(__int64 (__fastcall **)(_QWORD *))(*v18 + 64LL))(v18);
    *(_QWORD *)(v56 + 184) = v19 + v18[3] - v18[1] - v58;
    sub_390B9B0(a2, *(_QWORD **)(a1 + 8), v56, a3);
    *(_DWORD *)(v4 + 24) = v58;
    *(_DWORD *)(v4 + 28) = v59;
    sub_3919EA0(a1, v57);
    v20 = *(_DWORD *)(a1 + 272);
    if ( v20 )
    {
      v21 = *(_QWORD *)(v4 + 16);
      v22 = *(_QWORD *)(a1 + 256);
      v23 = (v20 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v24 = (_QWORD *)(v22 + 32LL * v23);
      v25 = *v24;
      if ( v21 == *v24 )
      {
        v26 = v24[1];
        v7 = 0xCCCCCCCCCCCCCCCDLL * ((v24[2] - v26) >> 3);
        goto LABEL_14;
      }
      v28 = 1;
      v29 = 0;
      while ( v25 != -8 )
      {
        if ( v29 || v25 != -16 )
          v24 = v29;
        v23 = (v20 - 1) & (v28 + v23);
        v48 = (__int64 *)(v22 + 32LL * v23);
        v25 = *v48;
        if ( v21 == *v48 )
        {
          v26 = v48[1];
          v7 = 0xCCCCCCCCCCCCCCCDLL * ((v48[2] - v26) >> 3);
          goto LABEL_14;
        }
        ++v28;
        v29 = v24;
        v24 = (_QWORD *)(v22 + 32LL * v23);
      }
      if ( !v29 )
        v29 = v24;
      v30 = *(_DWORD *)(a1 + 264);
      ++*(_QWORD *)(a1 + 248);
      v31 = v30 + 1;
      if ( 4 * v31 < 3 * v20 )
      {
        if ( v20 - *(_DWORD *)(a1 + 268) - v31 > v20 >> 3 )
          goto LABEL_23;
        sub_391A270(v49, v20);
        v41 = *(_DWORD *)(a1 + 272);
        if ( !v41 )
        {
LABEL_53:
          ++*(_DWORD *)(a1 + 264);
          BUG();
        }
        v42 = *(_QWORD *)(v4 + 16);
        v43 = v41 - 1;
        v44 = *(_QWORD *)(a1 + 256);
        v40 = 0;
        v45 = 1;
        LODWORD(v46) = v43 & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
        v31 = *(_DWORD *)(a1 + 264) + 1;
        v29 = (_QWORD *)(v44 + 32LL * (unsigned int)v46);
        v47 = *v29;
        if ( *v29 == v42 )
          goto LABEL_23;
        while ( v47 != -8 )
        {
          if ( v47 == -16 && !v40 )
            v40 = v29;
          v46 = v43 & (unsigned int)(v46 + v45);
          v29 = (_QWORD *)(v44 + 32 * v46);
          v47 = *v29;
          if ( v42 == *v29 )
            goto LABEL_23;
          ++v45;
        }
        goto LABEL_31;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 248);
    }
    sub_391A270(v49, 2 * v20);
    v33 = *(_DWORD *)(a1 + 272);
    if ( !v33 )
      goto LABEL_53;
    v34 = *(_QWORD *)(v4 + 16);
    v35 = v33 - 1;
    v36 = *(_QWORD *)(a1 + 256);
    LODWORD(v37) = v35 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
    v31 = *(_DWORD *)(a1 + 264) + 1;
    v29 = (_QWORD *)(v36 + 32LL * (unsigned int)v37);
    v38 = *v29;
    if ( *v29 == v34 )
      goto LABEL_23;
    v39 = 1;
    v40 = 0;
    while ( v38 != -8 )
    {
      if ( !v40 && v38 == -16 )
        v40 = v29;
      v37 = v35 & (unsigned int)(v37 + v39);
      v29 = (_QWORD *)(v36 + 32 * v37);
      v38 = *v29;
      if ( v34 == *v29 )
        goto LABEL_23;
      ++v39;
    }
LABEL_31:
    if ( v40 )
      v29 = v40;
LABEL_23:
    *(_DWORD *)(a1 + 264) = v31;
    if ( *v29 != -8 )
      --*(_DWORD *)(a1 + 268);
    v32 = *(_QWORD *)(v4 + 16);
    v26 = 0;
    v29[1] = 0;
    v29[2] = 0;
    *v29 = v32;
    v29[3] = 0;
LABEL_14:
    v27 = *(unsigned int *)(v4 + 24);
    v4 += 32;
    result = sub_39207C0(a1, v26, v7, v27);
  }
  return result;
}
