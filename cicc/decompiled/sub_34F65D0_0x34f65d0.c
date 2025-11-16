// Function: sub_34F65D0
// Address: 0x34f65d0
//
void __fastcall sub_34F65D0(__int64 a1, int a2, int a3)
{
  __int64 v3; // rax
  __int64 v7; // rdi
  int v8; // edx
  int v9; // edx
  __int64 v10; // r9
  __int64 v11; // r13
  unsigned int v12; // esi
  __int64 v13; // rdx
  __int64 v14; // r8
  unsigned int v15; // eax
  unsigned int *v16; // r14
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r10
  __int64 v21; // r9
  int v22; // eax
  __int64 v23; // rcx
  _DWORD *v24; // rdi
  int v25; // esi
  int v26; // eax
  _DWORD *v27; // rax
  __int64 v28; // r12
  _QWORD *v29; // rdi
  int v30; // r11d
  unsigned int *v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  int v37; // edi
  unsigned int v38; // r9d
  int i; // r10d
  int v40; // edi
  int v41; // eax
  int v42; // eax
  __int64 v43; // rsi
  unsigned int v44; // r14d
  int v45; // r10d
  __int64 v46; // [rsp+0h] [rbp-70h]
  __int64 v47; // [rsp+8h] [rbp-68h]
  __int64 v48; // [rsp+10h] [rbp-60h]
  __int64 v49; // [rsp+18h] [rbp-58h]
  unsigned __int64 *v50; // [rsp+20h] [rbp-50h] BYREF
  __int64 v51; // [rsp+28h] [rbp-48h]
  _BYTE *v52; // [rsp+30h] [rbp-40h] BYREF
  __int64 v53; // [rsp+38h] [rbp-38h]
  _BYTE v54[48]; // [rsp+40h] [rbp-30h] BYREF

  v3 = a3 & 0x7FFFFFFF;
  v7 = *(_QWORD *)(a1 + 40);
  v8 = *(_DWORD *)(*(_QWORD *)(v7 + 32) + 4 * v3);
  if ( v8 )
  {
    sub_300BF50(v7, a2, v8);
  }
  else
  {
    v9 = *(_DWORD *)(*(_QWORD *)(v7 + 56) + 4 * v3);
    if ( v9 == 0x7FFFFFFF )
      BUG();
    sub_300C140(v7, a2, v9);
  }
  v11 = *(_QWORD *)(a1 + 40);
  v12 = *(_DWORD *)(v11 + 128);
  v13 = *(_QWORD *)(v11 + 112);
  if ( v12 )
  {
    v14 = v12 - 1;
    v15 = v14 & (37 * a3);
    v16 = (unsigned int *)(v13 + 72LL * v15);
    v17 = *v16;
    if ( a3 == (_DWORD)v17 )
    {
LABEL_6:
      v46 = *((_QWORD *)v16 + 1);
      v47 = *((_QWORD *)v16 + 2);
      v48 = *((_QWORD *)v16 + 3);
      v18 = *((_QWORD *)v16 + 4);
      v50 = (unsigned __int64 *)&v52;
      v49 = v18;
      LODWORD(v18) = v16[12];
      v51 = 0;
      if ( (_DWORD)v18 )
        sub_34F5050((__int64)&v50, (__int64)(v16 + 10), v13, v17, v14, v10);
      v19 = v16[16];
      v53 = 0;
      v52 = v54;
      if ( (_DWORD)v19 )
        sub_34F4F70((__int64)&v52, (__int64)(v16 + 14), v19, v17, v14, v10);
      v12 = *(_DWORD *)(v11 + 128);
      v13 = *(_QWORD *)(v11 + 112);
      v20 = v11 + 104;
      if ( !v12 )
      {
        ++*(_QWORD *)(v11 + 104);
        goto LABEL_12;
      }
      v14 = v12 - 1;
    }
    else
    {
      v37 = *v16;
      v38 = v14 & (37 * a3);
      for ( i = 1; ; ++i )
      {
        if ( v37 == -1 )
          return;
        v38 = v14 & (i + v38);
        v37 = *(_DWORD *)(v13 + 72LL * v38);
        if ( a3 == v37 )
          break;
      }
      v40 = 1;
      while ( (_DWORD)v17 != -1 )
      {
        v10 = (unsigned int)(v40 + 1);
        v15 = v14 & (v40 + v15);
        v16 = (unsigned int *)(v13 + 72LL * v15);
        v17 = *v16;
        if ( a3 == (_DWORD)v17 )
          goto LABEL_6;
        ++v40;
      }
      v46 = 0;
      v47 = 0;
      v20 = v11 + 104;
      v48 = -1;
      v49 = -1;
      v50 = (unsigned __int64 *)&v52;
      v51 = 0;
      v52 = v54;
      v53 = 0;
    }
    v30 = 1;
    v24 = 0;
    v21 = (unsigned int)v14 & (37 * a2);
    v31 = (unsigned int *)(v13 + 72 * v21);
    v23 = *v31;
    if ( a2 == (_DWORD)v23 )
    {
LABEL_19:
      v29 = v31 + 2;
      v28 = (__int64)(v31 + 14);
LABEL_20:
      v32 = (__int64)(v29 + 4);
      *(_QWORD *)(v32 - 32) = v46;
      *(_QWORD *)(v32 - 24) = v47;
      *(_QWORD *)(v32 - 16) = v48;
      *(_QWORD *)(v32 - 8) = v49;
      sub_34F5050(v32, (__int64)&v50, v13, v23, v14, v21);
      sub_34F4F70(v28, (__int64)&v52, v33, v34, v35, v36);
      if ( v52 != v54 )
        _libc_free((unsigned __int64)v52);
      if ( v50 != (unsigned __int64 *)&v52 )
        _libc_free((unsigned __int64)v50);
      return;
    }
    while ( (_DWORD)v23 != -1 )
    {
      if ( (_DWORD)v23 == -2 && !v24 )
        v24 = v31;
      v21 = (unsigned int)v14 & (v30 + (_DWORD)v21);
      v31 = (unsigned int *)(v13 + 72LL * (unsigned int)v21);
      v23 = *v31;
      if ( a2 == (_DWORD)v23 )
        goto LABEL_19;
      ++v30;
    }
    if ( !v24 )
      v24 = v31;
    v41 = *(_DWORD *)(v11 + 120);
    ++*(_QWORD *)(v11 + 104);
    v26 = v41 + 1;
    if ( 4 * v26 < 3 * v12 )
    {
      v23 = v12 - (v26 + *(_DWORD *)(v11 + 124));
      v13 = v12 >> 3;
      if ( (unsigned int)v23 > (unsigned int)v13 )
      {
LABEL_14:
        *(_DWORD *)(v11 + 120) = v26;
        if ( *v24 != -1 )
          --*(_DWORD *)(v11 + 124);
        *v24 = a2;
        v27 = v24 + 18;
        v28 = (__int64)(v24 + 14);
        v29 = v24 + 2;
        *v29 = 0;
        v29[1] = 0;
        v29[2] = -1;
        v29[3] = -1;
        v29[4] = v28;
        v29[5] = 0;
        v29[6] = v27;
        v29[7] = 0;
        goto LABEL_20;
      }
      sub_2FB7300(v20, v12);
      v42 = *(_DWORD *)(v11 + 128);
      if ( v42 )
      {
        v13 = (unsigned int)(v42 - 1);
        v43 = *(_QWORD *)(v11 + 112);
        v21 = 1;
        v14 = 0;
        v44 = v13 & (37 * a2);
        v24 = (_DWORD *)(v43 + 72LL * v44);
        v23 = (unsigned int)*v24;
        v26 = *(_DWORD *)(v11 + 120) + 1;
        if ( a2 != (_DWORD)v23 )
        {
          while ( (_DWORD)v23 != -1 )
          {
            if ( !v14 && (_DWORD)v23 == -2 )
              v14 = (__int64)v24;
            v44 = v13 & (v21 + v44);
            v24 = (_DWORD *)(v43 + 72LL * v44);
            v23 = (unsigned int)*v24;
            if ( a2 == (_DWORD)v23 )
              goto LABEL_14;
            v21 = (unsigned int)(v21 + 1);
          }
          if ( v14 )
            v24 = (_DWORD *)v14;
        }
        goto LABEL_14;
      }
LABEL_66:
      ++*(_DWORD *)(v11 + 120);
      BUG();
    }
LABEL_12:
    sub_2FB7300(v20, 2 * v12);
    v22 = *(_DWORD *)(v11 + 128);
    if ( v22 )
    {
      v23 = (unsigned int)(v22 - 1);
      v14 = *(_QWORD *)(v11 + 112);
      v13 = (unsigned int)v23 & (37 * a2);
      v24 = (_DWORD *)(v14 + 72 * v13);
      v25 = *v24;
      v26 = *(_DWORD *)(v11 + 120) + 1;
      if ( *v24 != a2 )
      {
        v45 = 1;
        v21 = 0;
        while ( v25 != -1 )
        {
          if ( v25 == -2 && !v21 )
            v21 = (__int64)v24;
          v13 = (unsigned int)v23 & (v45 + (_DWORD)v13);
          v24 = (_DWORD *)(v14 + 72LL * (unsigned int)v13);
          v25 = *v24;
          if ( *v24 == a2 )
            goto LABEL_14;
          ++v45;
        }
        if ( v21 )
          v24 = (_DWORD *)v21;
      }
      goto LABEL_14;
    }
    goto LABEL_66;
  }
}
