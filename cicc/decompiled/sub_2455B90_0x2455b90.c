// Function: sub_2455B90
// Address: 0x2455b90
//
__int64 __fastcall sub_2455B90(__int64 *a1, __int64 a2)
{
  unsigned int v2; // r12d
  char *v3; // rax
  char *v4; // r8
  char *v5; // r9
  unsigned int v7; // edx
  char *v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdi
  int v12; // esi
  __int64 v13; // r8
  int v14; // esi
  unsigned int v15; // ecx
  __int64 *v16; // rax
  __int64 v17; // r10
  __int64 v18; // r15
  unsigned int v19; // eax
  __int64 v20; // r11
  unsigned int v21; // r13d
  unsigned int v22; // esi
  __int64 v23; // r9
  unsigned int v24; // r8d
  __int64 *v25; // rcx
  __int64 v26; // rdi
  unsigned int v27; // ecx
  int v28; // eax
  int v29; // r9d
  _QWORD *v30; // rax
  int v31; // ecx
  int v32; // edi
  int v33; // r10d
  int v34; // r10d
  __int64 v35; // r9
  unsigned int v36; // ecx
  __int64 v37; // rsi
  _QWORD *v38; // r8
  int v39; // r10d
  int v40; // r10d
  __int64 v41; // r9
  unsigned int v42; // ecx
  __int64 v43; // rsi
  unsigned int v44; // [rsp+0h] [rbp-100h]
  int v45; // [rsp+8h] [rbp-F8h]
  unsigned int v46; // [rsp+8h] [rbp-F8h]
  __int64 v47; // [rsp+8h] [rbp-F8h]
  char *v48; // [rsp+18h] [rbp-E8h]
  unsigned int v49; // [rsp+20h] [rbp-E0h]
  unsigned int v50; // [rsp+20h] [rbp-E0h]
  __int64 v51; // [rsp+20h] [rbp-E0h]
  int v52; // [rsp+20h] [rbp-E0h]
  int v53; // [rsp+20h] [rbp-E0h]
  char *v54; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v55; // [rsp+38h] [rbp-C8h]
  _BYTE v56[64]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE *v57; // [rsp+80h] [rbp-80h] BYREF
  __int64 v58; // [rsp+88h] [rbp-78h]
  _BYTE v59[112]; // [rsp+90h] [rbp-70h] BYREF

  v2 = 0;
  v54 = v56;
  v55 = 0x800000000LL;
  sub_D472F0(a2, (__int64)&v54);
  v3 = sub_2450F80(v54, &v54[8 * (unsigned int)v55]);
  if ( v5 != v3 )
    goto LABEL_2;
  if ( !(unsigned __int8)sub_D474B0(a2) || !sub_D4B130(a2) )
    goto LABEL_10;
  v58 = 0x800000000LL;
  v2 = -1;
  v57 = v59;
  sub_D46D90(a2, (__int64)&v57);
  if ( a1[23] )
    goto LABEL_8;
  v2 = qword_4FE6BA8;
  if ( (_DWORD)v58 == 1 )
    goto LABEL_8;
  if ( (unsigned int)qword_4FE69E8 < (unsigned int)v58 )
  {
    v2 = 0;
    goto LABEL_8;
  }
  v2 = qword_4FE6BA8;
  if ( (_BYTE)qword_4FE6908 )
    goto LABEL_8;
  v48 = &v54[8 * (unsigned int)v55];
  if ( v48 == v54 )
    goto LABEL_8;
  v7 = qword_4FE6BA8;
  v9 = v54;
  do
  {
    v10 = a1[22];
    v11 = *(_QWORD *)v9;
    v12 = *(_DWORD *)(v10 + 24);
    v13 = *(_QWORD *)(v10 + 8);
    if ( !v12 )
      goto LABEL_26;
    v14 = v12 - 1;
    v15 = v14 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v16 = (__int64 *)(v13 + 16LL * v15);
    v17 = *v16;
    if ( v11 == *v16 )
    {
LABEL_18:
      v18 = v16[1];
      if ( !v18 )
        goto LABEL_26;
      v49 = v7;
      v19 = sub_2455B90(a1, v16[1]);
      v20 = *a1;
      v7 = v49;
      v21 = v19;
      v22 = *(_DWORD *)(*a1 + 24);
      if ( v22 )
      {
        v23 = *(_QWORD *)(v20 + 8);
        v50 = ((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4);
        v24 = (v22 - 1) & v50;
        v25 = (__int64 *)(v23 + 152LL * v24);
        v26 = *v25;
        if ( v18 == *v25 )
        {
LABEL_21:
          v27 = *((_DWORD *)v25 + 4);
          if ( v21 < v27 )
            v21 = v27;
          v21 -= v27;
LABEL_24:
          if ( v7 > v21 )
            v7 = v21;
          goto LABEL_26;
        }
        v45 = 1;
        v30 = 0;
        while ( v26 != -4096 )
        {
          if ( v26 == -8192 && !v30 )
            v30 = v25;
          v24 = (v22 - 1) & (v45 + v24);
          v25 = (__int64 *)(v23 + 152LL * v24);
          v26 = *v25;
          if ( v18 == *v25 )
            goto LABEL_21;
          ++v45;
        }
        if ( !v30 )
          v30 = v25;
        v31 = *(_DWORD *)(v20 + 16);
        ++*(_QWORD *)v20;
        v32 = v31 + 1;
        if ( 4 * (v31 + 1) < 3 * v22 )
        {
          if ( v22 - *(_DWORD *)(v20 + 20) - v32 > v22 >> 3 )
            goto LABEL_39;
          v47 = v20;
          v44 = v7;
          sub_2455850(v20, v22);
          v20 = v47;
          v39 = *(_DWORD *)(v47 + 24);
          if ( !v39 )
          {
LABEL_69:
            ++*(_DWORD *)(v20 + 16);
            BUG();
          }
          v40 = v39 - 1;
          v41 = *(_QWORD *)(v47 + 8);
          v42 = v40 & v50;
          v32 = *(_DWORD *)(v47 + 16) + 1;
          v7 = v44;
          v30 = (_QWORD *)(v41 + 152LL * (v40 & v50));
          v43 = *v30;
          if ( *v30 == v18 )
            goto LABEL_39;
          v53 = 1;
          v38 = 0;
          while ( v43 != -4096 )
          {
            if ( v43 == -8192 && !v38 )
              v38 = v30;
            v42 = v40 & (v53 + v42);
            v30 = (_QWORD *)(v41 + 152LL * v42);
            v43 = *v30;
            if ( v18 == *v30 )
              goto LABEL_39;
            ++v53;
          }
          goto LABEL_47;
        }
      }
      else
      {
        ++*(_QWORD *)v20;
      }
      v51 = v20;
      v46 = v7;
      sub_2455850(v20, 2 * v22);
      v20 = v51;
      v33 = *(_DWORD *)(v51 + 24);
      if ( !v33 )
        goto LABEL_69;
      v34 = v33 - 1;
      v35 = *(_QWORD *)(v51 + 8);
      v36 = v34 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v32 = *(_DWORD *)(v51 + 16) + 1;
      v7 = v46;
      v30 = (_QWORD *)(v35 + 152LL * v36);
      v37 = *v30;
      if ( v18 == *v30 )
        goto LABEL_39;
      v52 = 1;
      v38 = 0;
      while ( v37 != -4096 )
      {
        if ( !v38 && v37 == -8192 )
          v38 = v30;
        v36 = v34 & (v52 + v36);
        v30 = (_QWORD *)(v35 + 152LL * v36);
        v37 = *v30;
        if ( v18 == *v30 )
          goto LABEL_39;
        ++v52;
      }
LABEL_47:
      if ( v38 )
        v30 = v38;
LABEL_39:
      *(_DWORD *)(v20 + 16) = v32;
      if ( *v30 != -4096 )
        --*(_DWORD *)(v20 + 20);
      *v30 = v18;
      v30[1] = v30 + 3;
      v30[2] = 0x800000000LL;
      goto LABEL_24;
    }
    v28 = 1;
    while ( v17 != -4096 )
    {
      v29 = v28 + 1;
      v15 = v14 & (v28 + v15);
      v16 = (__int64 *)(v13 + 16LL * v15);
      v17 = *v16;
      if ( v11 == *v16 )
        goto LABEL_18;
      v28 = v29;
    }
LABEL_26:
    v9 += 8;
  }
  while ( v48 != v9 );
  v2 = v7;
LABEL_8:
  if ( v57 != v59 )
    _libc_free((unsigned __int64)v57);
LABEL_10:
  v4 = v54;
LABEL_2:
  if ( v4 != v56 )
    _libc_free((unsigned __int64)v4);
  return v2;
}
