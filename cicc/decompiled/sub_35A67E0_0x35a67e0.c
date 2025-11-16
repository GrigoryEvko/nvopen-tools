// Function: sub_35A67E0
// Address: 0x35a67e0
//
__int64 __fastcall sub_35A67E0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 v9; // rsi
  __int64 v10; // rdi
  int v11; // ebx
  __int64 v12; // r13
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 i; // rdx
  __int64 v17; // r9
  bool v18; // cc
  __int64 v19; // r11
  __int64 *v20; // rbx
  __int64 *v21; // r13
  __int64 *v22; // r15
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rdi
  unsigned int v26; // ecx
  __int64 *v27; // rdx
  __int64 v28; // r10
  int v29; // r14d
  __int64 v30; // r12
  int v31; // r8d
  __int64 *v32; // rcx
  unsigned int v33; // r9d
  __int64 *v34; // rax
  __int64 v35; // rdi
  _DWORD *v36; // rax
  __int64 v37; // r14
  __int64 v38; // rdx
  __int64 v39; // rdi
  int v41; // eax
  int v42; // edx
  unsigned int v43; // edx
  __int64 v44; // rdi
  int v45; // r8d
  __int64 *v46; // r10
  int v47; // r8d
  unsigned int v48; // edx
  __int64 v49; // rdi
  __int64 *v50; // r12
  __int64 *v51; // rax
  __int64 v52; // rsi
  __int64 *v53; // rbx
  int v54; // ecx
  int v55; // edx
  int v56; // r8d
  __int64 v58; // [rsp+8h] [rbp-68h]
  int v59; // [rsp+14h] [rbp-5Ch]
  __int64 v60; // [rsp+18h] [rbp-58h]
  __int64 v61; // [rsp+20h] [rbp-50h] BYREF
  __int64 *v62; // [rsp+28h] [rbp-48h]
  __int64 v63; // [rsp+30h] [rbp-40h]
  unsigned int v64; // [rsp+38h] [rbp-38h]

  v7 = *(_QWORD *)a2;
  v8 = *(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v8 )
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
  *(_DWORD *)(a2 + 8) = 0;
  v11 = *(_DWORD *)(*a1 + 96) - 1;
  v12 = v11;
  if ( *(_DWORD *)(*a1 + 96) == 1 )
  {
    v61 = 0;
    v39 = 0;
    v62 = 0;
    v63 = 0;
    v64 = 0;
    return sub_C7D6A0(v39, v12, 8);
  }
  v13 = *(unsigned int *)(a2 + 12);
  v14 = 0;
  if ( v11 > v13 )
  {
    sub_359C370(a2, v11, v13, a4, a5, a6);
    v14 = 32LL * *(unsigned int *)(a2 + 8);
  }
  v15 = *(_QWORD *)a2 + v14;
  for ( i = *(_QWORD *)a2 + 32LL * v11; i != v15; v15 += 32 )
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = 0;
      *(_DWORD *)(v15 + 24) = 0;
      *(_QWORD *)(v15 + 8) = 0;
      *(_DWORD *)(v15 + 16) = 0;
      *(_DWORD *)(v15 + 20) = 0;
    }
  }
  v61 = 0;
  v62 = 0;
  *(_DWORD *)(a2 + 8) = v11;
  v17 = *a1;
  v63 = 0;
  v18 = *(_DWORD *)(v17 + 96) <= 1;
  v64 = 0;
  if ( v18 )
  {
    v39 = 0;
    v12 = 0;
    return sub_C7D6A0(v39, v12, 8);
  }
  v19 = 0;
  do
  {
    v20 = *(__int64 **)(v17 + 8);
    v21 = *(__int64 **)(v17 + 16);
    v59 = v19;
    if ( v20 == v21 )
      goto LABEL_26;
    v60 = v19;
    v22 = a1;
    v58 = 32 * v19;
    do
    {
      v23 = *v20;
      if ( *(_WORD *)(*v20 + 68) && *(_WORD *)(*v20 + 68) != 68 )
      {
        v24 = *(unsigned int *)(v17 + 88);
        v25 = *(_QWORD *)(v17 + 72);
        if ( (_DWORD)v24 )
        {
          v26 = (v24 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v27 = (__int64 *)(v25 + 16LL * v26);
          v28 = *v27;
          if ( v23 == *v27 )
          {
LABEL_18:
            if ( v27 != (__int64 *)(v25 + 16 * v24) )
            {
              v29 = *((_DWORD *)v27 + 2);
              if ( v29 > (int)v60 )
                goto LABEL_24;
LABEL_20:
              v30 = (__int64)sub_359A2A0((__int64)v22, v23);
              sub_359F7A0(v22, v30, *(_QWORD *)a2 + v58, 0);
              if ( v64 )
              {
                v31 = 1;
                v32 = 0;
                v33 = (v64 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
                v34 = &v62[2 * v33];
                v35 = *v34;
                if ( v30 == *v34 )
                {
LABEL_22:
                  v36 = v34 + 1;
LABEL_23:
                  v36[1] = v29;
                  *v36 = v59;
                  v37 = v22[10];
                  sub_2E31040((__int64 *)(v37 + 40), v30);
                  v38 = *(_QWORD *)(v37 + 48);
                  *(_QWORD *)(v30 + 8) = v37 + 48;
                  v38 &= 0xFFFFFFFFFFFFFFF8LL;
                  *(_QWORD *)v30 = v38 | *(_QWORD *)v30 & 7LL;
                  *(_QWORD *)(v38 + 8) = v30;
                  *(_QWORD *)(v37 + 48) = *(_QWORD *)(v37 + 48) & 7LL | v30;
                  v17 = *v22;
                  goto LABEL_24;
                }
                while ( v35 != -4096 )
                {
                  if ( !v32 && v35 == -8192 )
                    v32 = v34;
                  v33 = (v64 - 1) & (v31 + v33);
                  v34 = &v62[2 * v33];
                  v35 = *v34;
                  if ( v30 == *v34 )
                    goto LABEL_22;
                  ++v31;
                }
                if ( !v32 )
                  v32 = v34;
                ++v61;
                v41 = v63 + 1;
                if ( 4 * ((int)v63 + 1) < 3 * v64 )
                {
                  if ( v64 - HIDWORD(v63) - v41 > v64 >> 3 )
                    goto LABEL_39;
                  sub_35A57D0((__int64)&v61, v64);
                  if ( !v64 )
                  {
LABEL_83:
                    LODWORD(v63) = v63 + 1;
                    BUG();
                  }
                  v47 = 1;
                  v46 = 0;
                  v48 = (v64 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
                  v41 = v63 + 1;
                  v32 = &v62[2 * v48];
                  v49 = *v32;
                  if ( v30 == *v32 )
                    goto LABEL_39;
                  while ( v49 != -4096 )
                  {
                    if ( !v46 && v49 == -8192 )
                      v46 = v32;
                    v48 = (v64 - 1) & (v47 + v48);
                    v32 = &v62[2 * v48];
                    v49 = *v32;
                    if ( v30 == *v32 )
                      goto LABEL_39;
                    ++v47;
                  }
                  goto LABEL_58;
                }
              }
              else
              {
                ++v61;
              }
              sub_35A57D0((__int64)&v61, 2 * v64);
              if ( !v64 )
                goto LABEL_83;
              v43 = (v64 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
              v41 = v63 + 1;
              v32 = &v62[2 * v43];
              v44 = *v32;
              if ( v30 == *v32 )
                goto LABEL_39;
              v45 = 1;
              v46 = 0;
              while ( v44 != -4096 )
              {
                if ( v44 == -8192 && !v46 )
                  v46 = v32;
                v43 = (v64 - 1) & (v45 + v43);
                v32 = &v62[2 * v43];
                v44 = *v32;
                if ( v30 == *v32 )
                  goto LABEL_39;
                ++v45;
              }
LABEL_58:
              if ( v46 )
                v32 = v46;
LABEL_39:
              LODWORD(v63) = v41;
              if ( *v32 != -4096 )
                --HIDWORD(v63);
              *v32 = v30;
              v36 = v32 + 1;
              v32[1] = 0;
              goto LABEL_23;
            }
          }
          else
          {
            v42 = 1;
            while ( v28 != -4096 )
            {
              v56 = v42 + 1;
              v26 = (v24 - 1) & (v42 + v26);
              v27 = (__int64 *)(v25 + 16LL * v26);
              v28 = *v27;
              if ( v23 == *v27 )
                goto LABEL_18;
              v42 = v56;
            }
          }
        }
        v29 = -1;
        goto LABEL_20;
      }
LABEL_24:
      ++v20;
    }
    while ( v21 != v20 );
    v19 = v60;
    a1 = v22;
LABEL_26:
    ++v19;
  }
  while ( *(_DWORD *)(v17 + 96) - 1 > (int)v19 );
  v39 = (__int64)v62;
  v12 = 16LL * v64;
  if ( (_DWORD)v63 )
  {
    v50 = (__int64 *)((char *)v62 + v12);
    if ( v62 != (__int64 *)((char *)v62 + v12) )
    {
      v51 = v62;
      while ( 1 )
      {
        v52 = *v51;
        v53 = v51;
        if ( *v51 != -4096 && v52 != -8192 )
          break;
        v51 += 2;
        if ( v50 == v51 )
          return sub_C7D6A0(v39, v12, 8);
      }
      if ( v51 != v50 )
      {
        do
        {
          v54 = *((_DWORD *)v53 + 2);
          v55 = *((_DWORD *)v53 + 3);
          v53 += 2;
          sub_359CAC0(a1, v52, v55, v54, (_QWORD *)a2, 0);
          if ( v53 == v50 )
            break;
          while ( 1 )
          {
            v52 = *v53;
            if ( *v53 != -4096 && v52 != -8192 )
              break;
            v53 += 2;
            if ( v50 == v53 )
              goto LABEL_71;
          }
        }
        while ( v50 != v53 );
LABEL_71:
        v39 = (__int64)v62;
        v12 = 16LL * v64;
      }
    }
  }
  return sub_C7D6A0(v39, v12, 8);
}
