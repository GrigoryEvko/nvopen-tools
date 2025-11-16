// Function: sub_35A59B0
// Address: 0x35a59b0
//
__int64 __fastcall sub_35A59B0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rsi
  __int64 v10; // rdi
  int v11; // ebx
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 i; // rdx
  __int64 v16; // r9
  bool v17; // cc
  __int64 v18; // r13
  __int64 *v19; // rbx
  __int64 *v20; // r14
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rdi
  unsigned int v24; // ecx
  __int64 *v25; // rdx
  __int64 v26; // r8
  int v27; // r15d
  __int64 v28; // r12
  int v29; // r11d
  __int64 *v30; // rcx
  unsigned int v31; // r8d
  __int64 *v32; // rax
  __int64 v33; // rdi
  _DWORD *v34; // rax
  __int64 v35; // r15
  __int64 v36; // rdx
  int v38; // eax
  int v39; // edx
  int v40; // r11d
  unsigned int v41; // edx
  __int64 v42; // rdi
  int v43; // r11d
  __int64 *v44; // r9
  int v45; // r11d
  unsigned int v46; // edx
  __int64 v47; // rdi
  __int64 *v48; // rax
  __int64 *v49; // r12
  __int64 v50; // rsi
  __int64 *v51; // rbx
  int v52; // ecx
  int v53; // edx
  __int64 *v56; // [rsp+18h] [rbp-78h]
  __int64 v58; // [rsp+40h] [rbp-50h] BYREF
  __int64 *v59; // [rsp+48h] [rbp-48h]
  __int64 v60; // [rsp+50h] [rbp-40h]
  unsigned int v61; // [rsp+58h] [rbp-38h]

  v7 = *(_QWORD *)a3;
  v8 = *(_QWORD *)a3 + 32LL * *(unsigned int *)(a3 + 8);
  if ( *(_QWORD *)a3 != v8 )
  {
    do
    {
      v9 = *(unsigned int *)(v8 - 8);
      v10 = *(_QWORD *)(v8 - 24);
      v8 -= 32;
      sub_C7D6A0(v10, 8 * v9, 4);
    }
    while ( v7 != v8 );
  }
  *(_DWORD *)(a3 + 8) = 0;
  v11 = *(_DWORD *)(*a1 + 96) - 1;
  if ( *(_DWORD *)(*a1 + 96) == 1 )
  {
    v58 = 0;
    v59 = 0;
    v60 = 0;
    v61 = 0;
    goto LABEL_26;
  }
  v12 = *(unsigned int *)(a3 + 12);
  v13 = 0;
  if ( v11 > v12 )
  {
    sub_359C370(a3, v11, v12, a4, a5, a6);
    v13 = 32LL * *(unsigned int *)(a3 + 8);
  }
  v14 = *(_QWORD *)a3 + v13;
  for ( i = *(_QWORD *)a3 + 32LL * v11; i != v14; v14 += 32 )
  {
    if ( v14 )
    {
      *(_QWORD *)v14 = 0;
      *(_DWORD *)(v14 + 24) = 0;
      *(_QWORD *)(v14 + 8) = 0;
      *(_DWORD *)(v14 + 16) = 0;
      *(_DWORD *)(v14 + 20) = 0;
    }
  }
  v58 = 0;
  v59 = 0;
  *(_DWORD *)(a3 + 8) = v11;
  v16 = *a1;
  v60 = 0;
  v17 = *(_DWORD *)(v16 + 96) <= 1;
  v61 = 0;
  if ( !v17 )
  {
    v56 = a1;
    v18 = 0;
    while ( 1 )
    {
      v19 = *(__int64 **)(v16 + 8);
      v20 = *(__int64 **)(v16 + 16);
      if ( v19 != v20 )
        break;
LABEL_24:
      if ( *(_DWORD *)(v16 + 96) - 1 <= (int)++v18 )
      {
        a1 = v56;
        if ( (_DWORD)v60 )
        {
          v48 = v59;
          v49 = &v59[2 * v61];
          if ( v59 != v49 )
          {
            while ( 1 )
            {
              v50 = *v48;
              v51 = v48;
              if ( *v48 != -8192 && v50 != -4096 )
                break;
              v48 += 2;
              if ( v49 == v48 )
                goto LABEL_26;
            }
            if ( v49 != v48 )
            {
              do
              {
                v52 = *((_DWORD *)v51 + 2);
                v53 = *((_DWORD *)v51 + 3);
                v51 += 2;
                sub_359CAC0(v56, v50, v53, v52, (_QWORD *)a3, a2);
                if ( v51 == v49 )
                  break;
                while ( 1 )
                {
                  v50 = *v51;
                  if ( *v51 != -8192 && v50 != -4096 )
                    break;
                  v51 += 2;
                  if ( v49 == v51 )
                    goto LABEL_26;
                }
              }
              while ( v51 != v49 );
            }
          }
        }
        goto LABEL_26;
      }
    }
    while ( 1 )
    {
      v21 = *v19;
      if ( *(_WORD *)(*v19 + 68) )
      {
        if ( *(_WORD *)(*v19 + 68) != 68 )
        {
          v22 = *(unsigned int *)(v16 + 88);
          v23 = *(_QWORD *)(v16 + 72);
          if ( (_DWORD)v22 )
          {
            v24 = (v22 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
            v25 = (__int64 *)(v23 + 16LL * v24);
            v26 = *v25;
            if ( v21 != *v25 )
            {
              v39 = 1;
              while ( v26 != -4096 )
              {
                v40 = v39 + 1;
                v24 = (v22 - 1) & (v39 + v24);
                v25 = (__int64 *)(v23 + 16LL * v24);
                v26 = *v25;
                if ( v21 == *v25 )
                  goto LABEL_17;
                v39 = v40;
              }
              goto LABEL_23;
            }
LABEL_17:
            if ( v25 != (__int64 *)(v23 + 16 * v22) )
            {
              v27 = *((_DWORD *)v25 + 2);
              if ( v27 > (int)v18 )
                break;
            }
          }
        }
      }
LABEL_23:
      if ( v20 == ++v19 )
        goto LABEL_24;
    }
    v28 = (__int64)sub_359A2A0((__int64)v56, v21);
    sub_359F7A0(v56, v28, *(_QWORD *)a3 + 32 * v18, v27 - 1 == (_DWORD)v18);
    if ( v61 )
    {
      v29 = 1;
      v30 = 0;
      v31 = (v61 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v32 = &v59[2 * v31];
      v33 = *v32;
      if ( v28 == *v32 )
      {
LABEL_21:
        v34 = v32 + 1;
LABEL_22:
        v34[1] = v27;
        *v34 = v18;
        v35 = v56[12];
        sub_2E31040((__int64 *)(v35 + 40), v28);
        v36 = *(_QWORD *)(v35 + 48);
        *(_QWORD *)(v28 + 8) = v35 + 48;
        v36 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v28 = v36 | *(_QWORD *)v28 & 7LL;
        *(_QWORD *)(v36 + 8) = v28;
        *(_QWORD *)(v35 + 48) = *(_QWORD *)(v35 + 48) & 7LL | v28;
        v16 = *v56;
        goto LABEL_23;
      }
      while ( v33 != -4096 )
      {
        if ( v33 == -8192 && !v30 )
          v30 = v32;
        v31 = (v61 - 1) & (v29 + v31);
        v32 = &v59[2 * v31];
        v33 = *v32;
        if ( v28 == *v32 )
          goto LABEL_21;
        ++v29;
      }
      if ( !v30 )
        v30 = v32;
      ++v58;
      v38 = v60 + 1;
      if ( 4 * ((int)v60 + 1) < 3 * v61 )
      {
        if ( v61 - HIDWORD(v60) - v38 > v61 >> 3 )
          goto LABEL_37;
        sub_35A57D0((__int64)&v58, v61);
        if ( !v61 )
        {
LABEL_79:
          LODWORD(v60) = v60 + 1;
          BUG();
        }
        v45 = 1;
        v44 = 0;
        v46 = (v61 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v38 = v60 + 1;
        v30 = &v59[2 * v46];
        v47 = *v30;
        if ( v28 == *v30 )
          goto LABEL_37;
        while ( v47 != -4096 )
        {
          if ( v47 == -8192 && !v44 )
            v44 = v30;
          v46 = (v61 - 1) & (v45 + v46);
          v30 = &v59[2 * v46];
          v47 = *v30;
          if ( v28 == *v30 )
            goto LABEL_37;
          ++v45;
        }
        goto LABEL_57;
      }
    }
    else
    {
      ++v58;
    }
    sub_35A57D0((__int64)&v58, 2 * v61);
    if ( !v61 )
      goto LABEL_79;
    v41 = (v61 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
    v38 = v60 + 1;
    v30 = &v59[2 * v41];
    v42 = *v30;
    if ( v28 == *v30 )
      goto LABEL_37;
    v43 = 1;
    v44 = 0;
    while ( v42 != -4096 )
    {
      if ( !v44 && v42 == -8192 )
        v44 = v30;
      v41 = (v61 - 1) & (v43 + v41);
      v30 = &v59[2 * v41];
      v42 = *v30;
      if ( v28 == *v30 )
        goto LABEL_37;
      ++v43;
    }
LABEL_57:
    if ( v44 )
      v30 = v44;
LABEL_37:
    LODWORD(v60) = v38;
    if ( *v30 != -4096 )
      --HIDWORD(v60);
    *v30 = v28;
    v34 = v30 + 1;
    v30[1] = 0;
    goto LABEL_22;
  }
LABEL_26:
  sub_359A2D0((__int64)a1, a1[12], 0, a4, a1[13], a1[14]);
  return sub_C7D6A0((__int64)v59, 16LL * v61, 8);
}
