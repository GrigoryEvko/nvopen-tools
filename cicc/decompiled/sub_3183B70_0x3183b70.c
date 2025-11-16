// Function: sub_3183B70
// Address: 0x3183b70
//
unsigned __int8 *__fastcall sub_3183B70(unsigned __int8 *a1, __int64 a2)
{
  int v4; // eax
  __int64 v5; // rdi
  int v6; // ecx
  unsigned int v7; // eax
  unsigned __int8 **v8; // r12
  unsigned __int8 *v9; // rdx
  unsigned __int8 *v10; // rax
  unsigned __int8 *v11; // rax
  unsigned __int8 *v12; // r12
  unsigned __int8 *v13; // rax
  int v15; // esi
  unsigned __int8 *i; // rdi
  int v17; // edi
  unsigned __int8 *v18; // rax
  unsigned __int8 v19; // al
  __int64 v20; // r8
  int v21; // eax
  unsigned int v22; // esi
  __int64 v23; // r8
  int v24; // r10d
  unsigned int v25; // edi
  _QWORD *v26; // rdx
  unsigned __int8 **v27; // rax
  unsigned __int8 *v28; // rcx
  unsigned __int8 *v29; // rax
  _QWORD *v30; // r13
  unsigned __int8 *v31; // rax
  int v32; // ecx
  int v33; // ecx
  int v34; // eax
  int v35; // esi
  __int64 v36; // r8
  unsigned int v37; // edx
  unsigned __int8 *v38; // rdi
  int v39; // r10d
  unsigned __int8 **v40; // r9
  int v41; // eax
  int v42; // edx
  __int64 v43; // rdi
  int v44; // r9d
  unsigned __int8 **v45; // r8
  unsigned int v46; // r14d
  unsigned __int8 *v47; // rsi
  unsigned __int64 v48; // [rsp+0h] [rbp-50h] BYREF
  __int64 v49; // [rsp+8h] [rbp-48h]
  unsigned __int8 *v50; // [rsp+10h] [rbp-40h]
  unsigned __int64 v51; // [rsp+18h] [rbp-38h] BYREF
  __int64 v52; // [rsp+20h] [rbp-30h]
  unsigned __int8 *v53; // [rsp+28h] [rbp-28h]

  v4 = *(_DWORD *)(a2 + 24);
  v5 = *(_QWORD *)(a2 + 8);
  if ( v4 )
  {
    v6 = v4 - 1;
    v7 = (v4 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v8 = (unsigned __int8 **)(v5 + 56LL * v7);
    v9 = *v8;
    if ( *v8 == a1 )
    {
LABEL_3:
      v10 = v8[3];
      v48 = 4;
      v49 = 0;
      v50 = v10;
      if ( v10 != 0 && v10 + 4096 != 0 && v10 != (unsigned __int8 *)-8192LL )
        sub_BD6050(&v48, (unsigned __int64)v8[1] & 0xFFFFFFFFFFFFFFF8LL);
      v11 = v8[6];
      v51 = 6;
      v52 = 0;
      v53 = v11;
      if ( v11 != 0 && v11 + 4096 != 0 && v11 != (unsigned __int8 *)-8192LL )
        sub_BD6050(&v51, (unsigned __int64)v8[4] & 0xFFFFFFFFFFFFFFF8LL);
      if ( v50 )
      {
        v12 = v53;
        v13 = v53;
        if ( v53 )
          goto LABEL_11;
      }
      goto LABEL_21;
    }
    v15 = 1;
    while ( v9 != (unsigned __int8 *)-4096LL )
    {
      v7 = v6 & (v15 + v7);
      v8 = (unsigned __int8 **)(v5 + 56LL * v7);
      v9 = *v8;
      if ( *v8 == a1 )
        goto LABEL_3;
      ++v15;
    }
  }
  v48 = 4;
  v49 = 0;
  v50 = 0;
  v51 = 6;
  v52 = 0;
  v53 = 0;
LABEL_21:
  for ( i = a1; ; i = *(unsigned __int8 **)&v12[-32 * (*((_DWORD *)v12 + 1) & 0x7FFFFFF)] )
  {
    v18 = sub_98ACB0(i, 6u);
    v17 = 23;
    v12 = v18;
    v19 = *v18;
    if ( v19 > 0x1Cu )
    {
      if ( v19 != 85 )
      {
        v17 = 2 * (v19 != 34) + 21;
        goto LABEL_23;
      }
      v20 = *((_QWORD *)v12 - 4);
      v17 = 21;
      if ( v20 )
      {
        if ( !*(_BYTE *)v20 && *(_QWORD *)(v20 + 24) == *((_QWORD *)v12 + 10) )
          break;
      }
    }
LABEL_23:
    if ( !(unsigned __int8)sub_3108CA0(v17) )
      goto LABEL_31;
LABEL_24:
    ;
  }
  v21 = sub_3108960(*((_QWORD *)v12 - 4));
  if ( (unsigned __int8)sub_3108CA0(v21) )
    goto LABEL_24;
LABEL_31:
  v22 = *(_DWORD *)(a2 + 24);
  if ( v22 )
  {
    v23 = *(_QWORD *)(a2 + 8);
    v24 = 1;
    v25 = (v22 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v26 = (_QWORD *)(v23 + 56LL * v25);
    v27 = 0;
    v28 = (unsigned __int8 *)*v26;
    if ( (unsigned __int8 *)*v26 == a1 )
    {
LABEL_33:
      v29 = (unsigned __int8 *)v26[3];
      v30 = v26 + 1;
      if ( a1 != v29 )
      {
        if ( v29 != 0 && v29 + 4096 != 0 && v29 != (unsigned __int8 *)-8192LL )
          sub_BD60C0(v30);
        goto LABEL_37;
      }
      goto LABEL_40;
    }
    while ( v28 != (unsigned __int8 *)-4096LL )
    {
      if ( !v27 && v28 == (unsigned __int8 *)-8192LL )
        v27 = (unsigned __int8 **)v26;
      v25 = (v22 - 1) & (v24 + v25);
      v26 = (_QWORD *)(v23 + 56LL * v25);
      v28 = (unsigned __int8 *)*v26;
      if ( (unsigned __int8 *)*v26 == a1 )
        goto LABEL_33;
      ++v24;
    }
    v32 = *(_DWORD *)(a2 + 16);
    if ( !v27 )
      v27 = (unsigned __int8 **)v26;
    ++*(_QWORD *)a2;
    v33 = v32 + 1;
    if ( 4 * v33 < 3 * v22 )
    {
      if ( v22 - *(_DWORD *)(a2 + 20) - v33 > v22 >> 3 )
        goto LABEL_60;
      sub_3183850(a2, v22);
      v41 = *(_DWORD *)(a2 + 24);
      if ( v41 )
      {
        v42 = v41 - 1;
        v43 = *(_QWORD *)(a2 + 8);
        v44 = 1;
        v45 = 0;
        v46 = (v41 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v27 = (unsigned __int8 **)(v43 + 56LL * v46);
        v47 = *v27;
        v33 = *(_DWORD *)(a2 + 16) + 1;
        if ( *v27 != a1 )
        {
          while ( v47 != (unsigned __int8 *)-4096LL )
          {
            if ( v47 == (unsigned __int8 *)-8192LL && !v45 )
              v45 = v27;
            v46 = v42 & (v44 + v46);
            v27 = (unsigned __int8 **)(v43 + 56LL * v46);
            v47 = *v27;
            if ( *v27 == a1 )
              goto LABEL_60;
            ++v44;
          }
          if ( v45 )
            v27 = v45;
        }
        goto LABEL_60;
      }
LABEL_88:
      ++*(_DWORD *)(a2 + 16);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)a2;
  }
  sub_3183850(a2, 2 * v22);
  v34 = *(_DWORD *)(a2 + 24);
  if ( !v34 )
    goto LABEL_88;
  v35 = v34 - 1;
  v36 = *(_QWORD *)(a2 + 8);
  v37 = (v34 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v27 = (unsigned __int8 **)(v36 + 56LL * v37);
  v38 = *v27;
  v33 = *(_DWORD *)(a2 + 16) + 1;
  if ( *v27 != a1 )
  {
    v39 = 1;
    v40 = 0;
    while ( v38 != (unsigned __int8 *)-4096LL )
    {
      if ( !v40 && v38 == (unsigned __int8 *)-8192LL )
        v40 = v27;
      v37 = v35 & (v39 + v37);
      v27 = (unsigned __int8 **)(v36 + 56LL * v37);
      v38 = *v27;
      if ( *v27 == a1 )
        goto LABEL_60;
      ++v39;
    }
    if ( v40 )
      v27 = v40;
  }
LABEL_60:
  *(_DWORD *)(a2 + 16) = v33;
  if ( *v27 != (unsigned __int8 *)-4096LL )
    --*(_DWORD *)(a2 + 20);
  *v27 = a1;
  v30 = v27 + 1;
  v27[1] = (unsigned __int8 *)4;
  v27[2] = 0;
  v27[3] = 0;
  v27[4] = (unsigned __int8 *)6;
  v27[5] = 0;
  v27[6] = 0;
  if ( a1 )
  {
LABEL_37:
    v30[2] = a1;
    if ( a1 != 0 && a1 + 4096 != 0 && a1 != (unsigned __int8 *)-8192LL )
      sub_BD73F0((__int64)v30);
  }
LABEL_40:
  v31 = (unsigned __int8 *)v30[5];
  if ( v12 != v31 )
  {
    if ( v31 != 0 && v31 + 4096 != 0 && v31 != (unsigned __int8 *)-8192LL )
      sub_BD60C0(v30 + 3);
    v30[5] = v12;
    if ( v12 != (unsigned __int8 *)-4096LL && v12 != (unsigned __int8 *)-8192LL )
      sub_BD73F0((__int64)(v30 + 3));
  }
  v13 = v53;
LABEL_11:
  if ( v13 != 0 && v13 + 4096 != 0 && v13 != (unsigned __int8 *)-8192LL )
    sub_BD60C0(&v51);
  if ( v50 + 4096 != 0 && v50 != 0 && v50 != (unsigned __int8 *)-8192LL )
    sub_BD60C0(&v48);
  return v12;
}
