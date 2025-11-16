// Function: sub_350ABD0
// Address: 0x350abd0
//
void __fastcall sub_350ABD0(__int64 a1, int a2, int a3)
{
  __int64 v3; // r15
  unsigned int v6; // esi
  __int64 v7; // rdi
  unsigned int v8; // r8d
  int v9; // r10d
  unsigned int v10; // eax
  __int64 v11; // r11
  unsigned int *v12; // rbx
  int v13; // ecx
  __int64 v14; // r9
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // r8
  int v18; // eax
  int v19; // ecx
  __int64 v20; // rdi
  unsigned int v21; // edx
  _DWORD *v22; // rbx
  int v23; // esi
  int v24; // eax
  void **v25; // r12
  _DWORD *v26; // rax
  _QWORD *v27; // rbx
  int v28; // r10d
  unsigned int v29; // r9d
  _DWORD *v30; // rax
  int v31; // ecx
  int v32; // r8d
  size_t v33; // rdx
  int v34; // r8d
  size_t v35; // rdx
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rdi
  int v38; // eax
  int v39; // eax
  int v40; // ecx
  __int64 v41; // rdi
  _DWORD *v42; // r8
  int v43; // r9d
  unsigned int v44; // edx
  int v45; // esi
  int v46; // r9d
  int v47; // [rsp+Ch] [rbp-74h]
  int v48; // [rsp+Ch] [rbp-74h]
  __int64 v49; // [rsp+10h] [rbp-70h]
  __int64 v50; // [rsp+18h] [rbp-68h]
  __int64 v51; // [rsp+20h] [rbp-60h]
  __int64 v52; // [rsp+28h] [rbp-58h]
  void *dest; // [rsp+30h] [rbp-50h] BYREF
  __int64 v54; // [rsp+38h] [rbp-48h]
  void *v55; // [rsp+40h] [rbp-40h] BYREF
  __int64 v56; // [rsp+48h] [rbp-38h]
  _BYTE v57[48]; // [rsp+50h] [rbp-30h] BYREF

  v3 = a1 + 104;
  *(_DWORD *)(*(_QWORD *)(a1 + 80) + 4LL * (a2 & 0x7FFFFFFF)) = a3;
  v6 = *(_DWORD *)(a1 + 128);
  v7 = *(_QWORD *)(a1 + 112);
  if ( v6 )
  {
    v8 = v6 - 1;
    v9 = 1;
    v10 = (v6 - 1) & (37 * a3);
    LODWORD(v11) = v10;
    v12 = (unsigned int *)(v7 + 72LL * v10);
    v13 = *v12;
    v14 = *v12;
    if ( a3 == *v12 )
    {
LABEL_3:
      v15 = v12[12];
      v49 = *((_QWORD *)v12 + 1);
      v50 = *((_QWORD *)v12 + 2);
      v51 = *((_QWORD *)v12 + 3);
      v16 = *((_QWORD *)v12 + 4);
      dest = &v55;
      v52 = v16;
      v54 = 0;
      if ( (_DWORD)v15 && &dest != (void **)(v12 + 10) )
      {
        v47 = v15;
        sub_C8D5F0((__int64)&dest, &v55, (unsigned int)v15, 8u, v15, v14);
        v32 = v47;
        v33 = 8LL * v12[12];
        if ( v33 )
        {
          memcpy(dest, *((const void **)v12 + 5), v33);
          v32 = v47;
        }
        LODWORD(v54) = v32;
      }
      v17 = v12[16];
      v56 = 0;
      v55 = v57;
      if ( (_DWORD)v17 && &v55 != (void **)(v12 + 14) )
      {
        v48 = v17;
        sub_C8D5F0((__int64)&v55, v57, (unsigned int)v17, 8u, v17, v14);
        v34 = v48;
        v35 = 8LL * v12[16];
        if ( v35 )
        {
          memcpy(v55, *((const void **)v12 + 7), v35);
          v34 = v48;
        }
        LODWORD(v56) = v34;
      }
      v6 = *(_DWORD *)(a1 + 128);
      if ( !v6 )
      {
        ++*(_QWORD *)(a1 + 104);
        goto LABEL_7;
      }
      v7 = *(_QWORD *)(a1 + 112);
      v8 = v6 - 1;
    }
    else
    {
      while ( 1 )
      {
        if ( (_DWORD)v14 == -1 )
          return;
        v11 = v8 & ((_DWORD)v11 + v9);
        LODWORD(v14) = *(_DWORD *)(v7 + 72 * v11);
        if ( (_DWORD)v14 == a3 )
          break;
        ++v9;
      }
      v14 = 1;
      while ( v13 != -1 )
      {
        v10 = v8 & (v14 + v10);
        v12 = (unsigned int *)(v7 + 72LL * v10);
        v13 = *v12;
        if ( *v12 == a3 )
          goto LABEL_3;
        v14 = (unsigned int)(v14 + 1);
      }
      v49 = 0;
      v50 = 0;
      v51 = -1;
      v52 = -1;
      dest = &v55;
      v54 = 0;
      v55 = v57;
      v56 = 0;
    }
    v28 = 1;
    v22 = 0;
    v29 = v8 & (37 * a2);
    v30 = (_DWORD *)(v7 + 72LL * v29);
    v31 = *v30;
    if ( a2 == *v30 )
    {
LABEL_14:
      v27 = v30 + 2;
      v25 = (void **)(v30 + 14);
      goto LABEL_15;
    }
    while ( v31 != -1 )
    {
      if ( !v22 && v31 == -2 )
        v22 = v30;
      v29 = v8 & (v28 + v29);
      v30 = (_DWORD *)(v7 + 72LL * v29);
      v31 = *v30;
      if ( a2 == *v30 )
        goto LABEL_14;
      ++v28;
    }
    if ( !v22 )
      v22 = v30;
    v38 = *(_DWORD *)(a1 + 120);
    ++*(_QWORD *)(a1 + 104);
    v24 = v38 + 1;
    if ( 4 * v24 < 3 * v6 )
    {
      if ( v6 - (v24 + *(_DWORD *)(a1 + 124)) > v6 >> 3 )
        goto LABEL_9;
      sub_2FB7300(v3, v6);
      v39 = *(_DWORD *)(a1 + 128);
      if ( v39 )
      {
        v40 = v39 - 1;
        v41 = *(_QWORD *)(a1 + 112);
        v42 = 0;
        v43 = 1;
        v44 = (v39 - 1) & (37 * a2);
        v22 = (_DWORD *)(v41 + 72LL * v44);
        v45 = *v22;
        v24 = *(_DWORD *)(a1 + 120) + 1;
        if ( a2 == *v22 )
          goto LABEL_9;
        while ( v45 != -1 )
        {
          if ( v45 == -2 && !v42 )
            v42 = v22;
          v44 = v40 & (v43 + v44);
          v22 = (_DWORD *)(v41 + 72LL * v44);
          v45 = *v22;
          if ( a2 == *v22 )
            goto LABEL_9;
          ++v43;
        }
        goto LABEL_59;
      }
LABEL_76:
      ++*(_DWORD *)(a1 + 120);
      BUG();
    }
LABEL_7:
    sub_2FB7300(v3, 2 * v6);
    v18 = *(_DWORD *)(a1 + 128);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a1 + 112);
      v21 = (v18 - 1) & (37 * a2);
      v22 = (_DWORD *)(v20 + 72LL * v21);
      v23 = *v22;
      v24 = *(_DWORD *)(a1 + 120) + 1;
      if ( a2 == *v22 )
      {
LABEL_9:
        *(_DWORD *)(a1 + 120) = v24;
        if ( *v22 != -1 )
          --*(_DWORD *)(a1 + 124);
        v25 = (void **)(v22 + 14);
        v26 = v22 + 18;
        *v22 = a2;
        v27 = v22 + 2;
        *v27 = 0;
        v27[1] = 0;
        v27[2] = -1;
        v27[3] = -1;
        v27[4] = v25;
        v27[5] = 0;
        v27[6] = v26;
        v27[7] = 0;
LABEL_15:
        *v27 = v49;
        v27[1] = v50;
        v27[2] = v51;
        v27[3] = v52;
        if ( &dest != v27 + 4 )
        {
          if ( (_DWORD)v54 )
          {
            v37 = v27[4];
            if ( v25 != (void **)v37 )
              _libc_free(v37);
            v27[4] = dest;
            dest = &v55;
            v27[5] = v54;
            v54 = 0;
          }
          else
          {
            *((_DWORD *)v27 + 10) = 0;
          }
        }
        if ( v25 != &v55 )
        {
          if ( (_DWORD)v56 )
          {
            v36 = v27[6];
            if ( (_QWORD *)v36 != v27 + 8 )
              _libc_free(v36);
            v27[6] = v55;
            v27[7] = v56;
            goto LABEL_23;
          }
          *((_DWORD *)v27 + 14) = 0;
        }
        if ( v55 != v57 )
          _libc_free((unsigned __int64)v55);
LABEL_23:
        if ( dest != &v55 )
          _libc_free((unsigned __int64)dest);
        return;
      }
      v46 = 1;
      v42 = 0;
      while ( v23 != -1 )
      {
        if ( v23 == -2 && !v42 )
          v42 = v22;
        v21 = v19 & (v46 + v21);
        v22 = (_DWORD *)(v20 + 72LL * v21);
        v23 = *v22;
        if ( a2 == *v22 )
          goto LABEL_9;
        ++v46;
      }
LABEL_59:
      if ( v42 )
        v22 = v42;
      goto LABEL_9;
    }
    goto LABEL_76;
  }
}
