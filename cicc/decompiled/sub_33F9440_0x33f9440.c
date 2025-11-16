// Function: sub_33F9440
// Address: 0x33f9440
//
void __fastcall sub_33F9440(__int64 a1, _QWORD *a2, char a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 *v12; // rdx
  __int64 *v13; // r13
  unsigned __int64 v14; // rcx
  __int64 *v15; // rbx
  __int64 v16; // r14
  __int64 *v17; // rdi
  __int64 *v18; // r11
  __int64 *v19; // rbx
  __int64 v20; // r8
  __int64 v21; // rdi
  unsigned int v22; // ecx
  __int64 *v23; // r14
  __int64 v24; // rdx
  __int64 v25; // rax
  _QWORD *v26; // r13
  __int64 v27; // r13
  unsigned int v28; // esi
  int v29; // eax
  int v30; // esi
  __int64 v31; // rdi
  __int64 v32; // rdx
  int v33; // ecx
  _QWORD *v34; // rax
  __int64 v35; // r9
  __int64 v36; // rax
  int v37; // edi
  int v38; // eax
  int v39; // edx
  _QWORD *v40; // r9
  int v41; // r10d
  __int64 v42; // r15
  __int64 v43; // rdi
  __int64 v44; // rsi
  int v45; // r14d
  _QWORD *v46; // r10
  __int64 v47; // [rsp+8h] [rbp-88h]
  __int64 v48; // [rsp+8h] [rbp-88h]
  __int64 v49; // [rsp+8h] [rbp-88h]
  __int64 *v50; // [rsp+10h] [rbp-80h]
  __int64 *v51; // [rsp+10h] [rbp-80h]
  int v52; // [rsp+10h] [rbp-80h]
  __int64 *v53; // [rsp+10h] [rbp-80h]
  __int64 *v54; // [rsp+20h] [rbp-70h] BYREF
  __int64 v55; // [rsp+28h] [rbp-68h]
  _BYTE v56[96]; // [rsp+30h] [rbp-60h] BYREF

  if ( a3 )
  {
    v7 = *(unsigned int *)(a1 + 376);
    if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 380) )
    {
      sub_C8D5F0(a1 + 368, (const void *)(a1 + 384), v7 + 1, 8u, a5, a6);
      v7 = *(unsigned int *)(a1 + 376);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 368) + 8 * v7) = a2;
    ++*(_DWORD *)(a1 + 376);
  }
  else
  {
    v36 = *(unsigned int *)(a1 + 104);
    if ( v36 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 108) )
    {
      sub_C8D5F0(a1 + 96, (const void *)(a1 + 112), v36 + 1, 8u, a5, a6);
      v36 = *(unsigned int *)(a1 + 104);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v36) = a2;
    ++*(_DWORD *)(a1 + 104);
  }
  v8 = a2[1];
  v54 = (__int64 *)v56;
  v55 = 0x600000000LL;
  v9 = v8 + 24LL * *a2;
  if ( v8 == v9 )
  {
    v12 = (__int64 *)a2[3];
    v13 = &v12[a2[2]];
    if ( v13 == v12 )
      return;
    v14 = 6;
    v10 = 0;
  }
  else
  {
    v10 = 0;
    do
    {
      while ( *(_DWORD *)v8 )
      {
        v8 += 24;
        if ( v9 == v8 )
          goto LABEL_12;
      }
      v11 = *(_QWORD *)(v8 + 8);
      if ( v10 + 1 > (unsigned __int64)HIDWORD(v55) )
      {
        sub_C8D5F0((__int64)&v54, v56, v10 + 1, 8u, a5, a6);
        v10 = (unsigned int)v55;
      }
      v8 += 24;
      v54[v10] = v11;
      v10 = (unsigned int)(v55 + 1);
      LODWORD(v55) = v55 + 1;
    }
    while ( v9 != v8 );
LABEL_12:
    v12 = (__int64 *)a2[3];
    v13 = &v12[a2[2]];
    if ( v12 == v13 )
      goto LABEL_19;
    v14 = HIDWORD(v55);
  }
  v15 = v12;
  while ( 1 )
  {
    v16 = *v15;
    if ( v10 + 1 > v14 )
    {
      sub_C8D5F0((__int64)&v54, v56, v10 + 1, 8u, a5, a6);
      v10 = (unsigned int)v55;
    }
    ++v15;
    v54[v10] = v16;
    v10 = (unsigned int)(v55 + 1);
    LODWORD(v55) = v55 + 1;
    if ( v13 == v15 )
      break;
    v14 = HIDWORD(v55);
  }
LABEL_19:
  v17 = v54;
  v18 = &v54[v10];
  if ( v18 != v54 )
  {
    v19 = v54;
    v20 = (__int64)a2;
    while ( 1 )
    {
      v27 = *v19;
      if ( !*v19 )
        goto LABEL_25;
      v28 = *(_DWORD *)(a1 + 712);
      if ( v28 )
      {
        v21 = *(_QWORD *)(a1 + 696);
        v22 = (v28 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v23 = (__int64 *)(v21 + 40LL * v22);
        v24 = *v23;
        if ( v27 == *v23 )
        {
LABEL_22:
          v25 = *((unsigned int *)v23 + 4);
          v26 = v23 + 1;
          if ( v25 + 1 > (unsigned __int64)*((unsigned int *)v23 + 5) )
          {
            v51 = v18;
            v48 = v20;
            sub_C8D5F0((__int64)(v23 + 1), v23 + 3, v25 + 1, 8u, v20, a1 + 688);
            v25 = *((unsigned int *)v23 + 4);
            v20 = v48;
            v18 = v51;
          }
          goto LABEL_24;
        }
        v52 = 1;
        v34 = 0;
        while ( v24 != -4096 )
        {
          if ( !v34 && v24 == -8192 )
            v34 = v23;
          v22 = (v28 - 1) & (v52 + v22);
          v23 = (__int64 *)(v21 + 40LL * v22);
          v24 = *v23;
          if ( v27 == *v23 )
            goto LABEL_22;
          ++v52;
        }
        v37 = *(_DWORD *)(a1 + 704);
        if ( !v34 )
          v34 = v23;
        ++*(_QWORD *)(a1 + 688);
        v33 = v37 + 1;
        if ( 4 * (v37 + 1) < 3 * v28 )
        {
          if ( v28 - *(_DWORD *)(a1 + 708) - v33 <= v28 >> 3 )
          {
            v49 = v20;
            v53 = v18;
            sub_33F9120(a1 + 688, v28);
            v38 = *(_DWORD *)(a1 + 712);
            if ( !v38 )
            {
LABEL_75:
              ++*(_DWORD *)(a1 + 704);
              BUG();
            }
            v39 = v38 - 1;
            v40 = 0;
            v18 = v53;
            v41 = 1;
            LODWORD(v42) = (v38 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
            v43 = *(_QWORD *)(a1 + 696);
            v20 = v49;
            v33 = *(_DWORD *)(a1 + 704) + 1;
            v34 = (_QWORD *)(v43 + 40LL * (unsigned int)v42);
            v44 = *v34;
            if ( v27 != *v34 )
            {
              while ( v44 != -4096 )
              {
                if ( v44 == -8192 && !v40 )
                  v40 = v34;
                v42 = v39 & (unsigned int)(v42 + v41);
                v34 = (_QWORD *)(v43 + 40 * v42);
                v44 = *v34;
                if ( v27 == *v34 )
                  goto LABEL_31;
                ++v41;
              }
              if ( v40 )
                v34 = v40;
            }
          }
          goto LABEL_31;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 688);
      }
      v47 = v20;
      v50 = v18;
      sub_33F9120(a1 + 688, 2 * v28);
      v29 = *(_DWORD *)(a1 + 712);
      if ( !v29 )
        goto LABEL_75;
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 696);
      v18 = v50;
      v20 = v47;
      LODWORD(v32) = (v29 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v33 = *(_DWORD *)(a1 + 704) + 1;
      v34 = (_QWORD *)(v31 + 40LL * (unsigned int)v32);
      v35 = *v34;
      if ( v27 != *v34 )
      {
        v45 = 1;
        v46 = 0;
        while ( v35 != -4096 )
        {
          if ( !v46 && v35 == -8192 )
            v46 = v34;
          v32 = v30 & (unsigned int)(v32 + v45);
          v34 = (_QWORD *)(v31 + 40 * v32);
          v35 = *v34;
          if ( v27 == *v34 )
            goto LABEL_31;
          ++v45;
        }
        if ( v46 )
          v34 = v46;
      }
LABEL_31:
      *(_DWORD *)(a1 + 704) = v33;
      if ( *v34 != -4096 )
        --*(_DWORD *)(a1 + 708);
      *v34 = v27;
      v26 = v34 + 1;
      v34[1] = v34 + 3;
      v34[2] = 0x200000000LL;
      v25 = 0;
LABEL_24:
      *(_QWORD *)(*v26 + 8 * v25) = v20;
      ++*((_DWORD *)v26 + 2);
LABEL_25:
      if ( v18 == ++v19 )
      {
        v17 = v54;
        break;
      }
    }
  }
  if ( v17 != (__int64 *)v56 )
    _libc_free((unsigned __int64)v17);
}
