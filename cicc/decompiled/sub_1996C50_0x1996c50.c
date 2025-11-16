// Function: sub_1996C50
// Address: 0x1996c50
//
void __fastcall sub_1996C50(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v4; // r13
  void *v5; // rdi
  unsigned int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // r14
  __int64 v10; // rsi
  __int64 *v11; // rax
  __int64 *v12; // r15
  __int64 *v13; // rbx
  __int64 *v14; // r9
  __int64 *v15; // r10
  __int64 v16; // rsi
  __int64 *v17; // rdi
  unsigned int v18; // r8d
  __int64 *v19; // rax
  __int64 *v20; // rcx
  unsigned __int64 v21; // rdi
  __int64 *v22; // rcx
  __int64 *v23; // r12
  __int64 *v24; // rax
  __int64 v25; // r14
  __int64 *v26; // rbx
  __int64 *v27; // rdi
  unsigned int v28; // r9d
  __int64 *v29; // rcx
  __int64 v30; // rax
  __int64 *v31; // r15
  __int64 *v32; // r12
  __int64 v33; // rbx
  _QWORD *v34; // rcx
  _QWORD *v35; // rax
  _QWORD *v36; // r13
  __int64 v37; // rax
  __int64 *v38; // rax
  unsigned int v39; // esi
  __int64 v40; // rdi
  unsigned int v41; // ecx
  __int64 *v42; // rax
  __int64 v43; // r9
  unsigned __int64 v44; // rdi
  __int64 v45; // rcx
  __int64 v46; // rsi
  int v47; // eax
  _QWORD *v48; // rcx
  int v49; // edx
  __int64 v50; // [rsp+8h] [rbp-A8h]
  __int64 v53; // [rsp+28h] [rbp-88h]
  __int64 v54; // [rsp+28h] [rbp-88h]
  __int64 v55; // [rsp+30h] [rbp-80h] BYREF
  __int64 *v56; // [rsp+38h] [rbp-78h]
  __int64 *v57; // [rsp+40h] [rbp-70h]
  int v58; // [rsp+48h] [rbp-68h]
  int v59; // [rsp+4Ch] [rbp-64h]
  char v60[88]; // [rsp+58h] [rbp-58h] BYREF

  v3 = a1;
  v4 = a1 + 1912;
  sub_16CCEE0(&v55, (__int64)v60, 4, a1 + 1912);
  ++*(_QWORD *)(a1 + 1912);
  v5 = *(void **)(a1 + 1928);
  if ( v5 == *(void **)(v3 + 1920) )
    goto LABEL_6;
  v6 = 4 * (*(_DWORD *)(v3 + 1940) - *(_DWORD *)(v3 + 1944));
  v7 = *(unsigned int *)(v3 + 1936);
  if ( v6 < 0x20 )
    v6 = 32;
  if ( (unsigned int)v7 <= v6 )
  {
    memset(v5, -1, 8 * v7);
LABEL_6:
    *(_QWORD *)(v3 + 1940) = 0;
    goto LABEL_7;
  }
  sub_16CC920(v4);
LABEL_7:
  v8 = *(_QWORD *)(v3 + 744);
  v9 = v3;
  v53 = v8 + 96LL * *(unsigned int *)(v3 + 752);
  if ( v53 != v8 )
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(v8 + 80);
      if ( !v10 )
        goto LABEL_11;
      v11 = *(__int64 **)(v9 + 1920);
      if ( *(__int64 **)(v9 + 1928) != v11 )
        break;
      v27 = &v11[*(unsigned int *)(v9 + 1940)];
      v28 = *(_DWORD *)(v9 + 1940);
      if ( v11 == v27 )
      {
LABEL_80:
        if ( v28 >= *(_DWORD *)(v9 + 1936) )
          break;
        *(_DWORD *)(v9 + 1940) = v28 + 1;
        *v27 = v10;
        ++*(_QWORD *)(v9 + 1912);
      }
      else
      {
        v29 = 0;
        while ( v10 != *v11 )
        {
          if ( *v11 == -2 )
            v29 = v11;
          if ( v27 == ++v11 )
          {
            if ( !v29 )
              goto LABEL_80;
            *v29 = v10;
            --*(_DWORD *)(v9 + 1944);
            ++*(_QWORD *)(v9 + 1912);
            break;
          }
        }
      }
LABEL_11:
      v12 = *(__int64 **)(v8 + 32);
      v13 = &v12[*(unsigned int *)(v8 + 40)];
      if ( v12 != v13 )
      {
        v14 = *(__int64 **)(v9 + 1928);
        v15 = *(__int64 **)(v9 + 1920);
        do
        {
LABEL_15:
          v16 = *v12;
          if ( v14 != v15 )
            goto LABEL_13;
          v17 = &v14[*(unsigned int *)(v9 + 1940)];
          v18 = *(_DWORD *)(v9 + 1940);
          if ( v17 != v14 )
          {
            v19 = v14;
            v20 = 0;
            while ( v16 != *v19 )
            {
              if ( *v19 == -2 )
                v20 = v19;
              if ( v17 == ++v19 )
              {
                if ( !v20 )
                  goto LABEL_35;
                ++v12;
                *v20 = v16;
                v14 = *(__int64 **)(v9 + 1928);
                --*(_DWORD *)(v9 + 1944);
                v15 = *(__int64 **)(v9 + 1920);
                ++*(_QWORD *)(v9 + 1912);
                if ( v13 != v12 )
                  goto LABEL_15;
                goto LABEL_24;
              }
            }
            goto LABEL_14;
          }
LABEL_35:
          if ( v18 < *(_DWORD *)(v9 + 1936) )
          {
            *(_DWORD *)(v9 + 1940) = v18 + 1;
            *v17 = v16;
            v15 = *(__int64 **)(v9 + 1920);
            ++*(_QWORD *)(v9 + 1912);
            v14 = *(__int64 **)(v9 + 1928);
          }
          else
          {
LABEL_13:
            sub_16CCBA0(v4, v16);
            v14 = *(__int64 **)(v9 + 1928);
            v15 = *(__int64 **)(v9 + 1920);
          }
LABEL_14:
          ++v12;
        }
        while ( v13 != v12 );
      }
LABEL_24:
      v8 += 96;
      if ( v53 == v8 )
      {
        v3 = v9;
        goto LABEL_26;
      }
    }
    sub_16CCBA0(v4, v10);
    goto LABEL_11;
  }
LABEL_26:
  v21 = (unsigned __int64)v57;
  v22 = v56;
  if ( v57 == v56 )
    v23 = &v57[v59];
  else
    v23 = &v57[v58];
  if ( v57 != v23 )
  {
    v24 = v57;
    while ( 1 )
    {
      v25 = *v24;
      v26 = v24;
      if ( (unsigned __int64)*v24 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v23 == ++v24 )
        goto LABEL_32;
    }
    if ( v23 != v24 )
    {
      v54 = v4;
      v50 = ~(1LL << a2);
      v30 = v3;
      v31 = v23;
      v32 = v26;
      v33 = v30;
      while ( 1 )
      {
        v34 = *(_QWORD **)(v33 + 1928);
        v35 = *(_QWORD **)(v33 + 1920);
        if ( v34 == v35 )
        {
          v36 = &v35[*(unsigned int *)(v33 + 1940)];
          if ( v35 == v36 )
          {
            v48 = *(_QWORD **)(v33 + 1920);
          }
          else
          {
            do
            {
              if ( v25 == *v35 )
                break;
              ++v35;
            }
            while ( v36 != v35 );
            v48 = v36;
          }
LABEL_65:
          while ( v48 != v35 )
          {
            if ( *v35 < 0xFFFFFFFFFFFFFFFELL )
              goto LABEL_51;
            ++v35;
          }
          if ( v36 != v35 )
            goto LABEL_52;
        }
        else
        {
          v36 = &v34[*(unsigned int *)(v33 + 1936)];
          v35 = sub_16CC9F0(v54, v25);
          if ( v25 == *v35 )
          {
            v45 = *(_QWORD *)(v33 + 1928);
            if ( v45 == *(_QWORD *)(v33 + 1920) )
              v46 = *(unsigned int *)(v33 + 1940);
            else
              v46 = *(unsigned int *)(v33 + 1936);
            v48 = (_QWORD *)(v45 + 8 * v46);
            goto LABEL_65;
          }
          v37 = *(_QWORD *)(v33 + 1928);
          if ( v37 == *(_QWORD *)(v33 + 1920) )
          {
            v35 = (_QWORD *)(v37 + 8LL * *(unsigned int *)(v33 + 1940));
            v48 = v35;
            goto LABEL_65;
          }
          v35 = (_QWORD *)(v37 + 8LL * *(unsigned int *)(v33 + 1936));
LABEL_51:
          if ( v36 != v35 )
            goto LABEL_52;
        }
        v39 = *(_DWORD *)(a3 + 24);
        v40 = *(_QWORD *)(a3 + 8);
        if ( v39 )
        {
          v41 = (v39 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          v42 = (__int64 *)(v40 + 16LL * v41);
          v43 = *v42;
          if ( *v42 == v25 )
          {
LABEL_69:
            v44 = v42[1];
            if ( (v44 & 1) != 0 )
              goto LABEL_70;
            goto LABEL_77;
          }
          v47 = 1;
          while ( v43 != -8 )
          {
            v49 = v47 + 1;
            v41 = (v39 - 1) & (v47 + v41);
            v42 = (__int64 *)(v40 + 16LL * v41);
            v43 = *v42;
            if ( v25 == *v42 )
              goto LABEL_69;
            v47 = v49;
          }
        }
        v42 = (__int64 *)(v40 + 16LL * v39);
        v44 = v42[1];
        if ( (v44 & 1) != 0 )
        {
LABEL_70:
          v42[1] = 2 * ((v44 >> 58 << 57) | v50 & (v44 >> 1) & ~(-1LL << (v44 >> 58))) + 1;
          goto LABEL_52;
        }
LABEL_77:
        *(_QWORD *)(*(_QWORD *)v44 + 8LL * (a2 >> 6)) &= v50;
LABEL_52:
        v38 = v32 + 1;
        if ( v32 + 1 != v31 )
        {
          while ( 1 )
          {
            v25 = *v38;
            v32 = v38;
            if ( (unsigned __int64)*v38 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v31 == ++v38 )
              goto LABEL_55;
          }
          if ( v31 != v38 )
            continue;
        }
LABEL_55:
        v21 = (unsigned __int64)v57;
        v22 = v56;
        break;
      }
    }
  }
LABEL_32:
  if ( (__int64 *)v21 != v22 )
    _libc_free(v21);
}
