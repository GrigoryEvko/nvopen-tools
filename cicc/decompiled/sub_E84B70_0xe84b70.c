// Function: sub_E84B70
// Address: 0xe84b70
//
__int64 __fastcall sub_E84B70(__int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // r9
  __int64 v3; // r12
  __int64 *v4; // rbx
  __int64 v5; // rax
  __int64 *v6; // r15
  __int64 v7; // r12
  char v8; // al
  void *v9; // r13
  void *v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  _QWORD *v14; // r12
  __int64 v15; // rbx
  __int64 v16; // r13
  __int64 v17; // rsi
  _QWORD *v18; // r14
  unsigned int v19; // edx
  _QWORD *v20; // rax
  _QWORD *v21; // r9
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 v24; // rax
  _QWORD *v25; // rbx
  __int64 v26; // rdi
  _QWORD *v27; // r13
  __int64 v28; // r12
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // r12
  unsigned __int64 v34; // r12
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rax
  _QWORD *v38; // r12
  size_t v39; // rbx
  __int64 v40; // rdi
  int v42; // eax
  int v43; // r10d
  int v44; // r11d
  unsigned int v45; // ecx
  __int64 *v46; // rdx
  __int64 *v47; // rax
  void *v48; // r10
  _QWORD *v49; // rax
  __int64 v50; // rbx
  __int64 v51; // r8
  __int64 v52; // r9
  _QWORD *v53; // rbx
  __int64 v54; // rax
  __int64 v55; // rcx
  int v56; // edx
  __int64 v57; // r14
  int v58; // ecx
  void *v59; // rsi
  int v60; // edi
  __int64 *v61; // rsi
  _QWORD *v63; // [rsp+8h] [rbp-58h]
  __int64 *v64; // [rsp+8h] [rbp-58h]
  __int64 v65; // [rsp+10h] [rbp-50h] BYREF
  __int64 v66; // [rsp+18h] [rbp-48h]
  __int64 v67; // [rsp+20h] [rbp-40h]
  unsigned int v68; // [rsp+28h] [rbp-38h]

  sub_E8ACF0(a1, *(_QWORD *)(*(_QWORD *)(a1 + 296) + 8LL));
  v3 = *(_QWORD *)(a1 + 296);
  v65 = 0;
  v66 = 0;
  v4 = *(__int64 **)(v3 + 56);
  v5 = *(unsigned int *)(v3 + 64);
  v67 = 0;
  v68 = 0;
  v6 = &v4[v5];
  if ( v6 != v4 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v7 = *v4;
        v8 = *(_BYTE *)(*v4 + 8);
        if ( (v8 & 2) != 0 && (*(_BYTE *)(v7 + 9) & 8) == 0 )
          goto LABEL_6;
        v9 = *(void **)v7;
        if ( *(_QWORD *)v7 )
          break;
        if ( (*(_BYTE *)(v7 + 9) & 0x70) != 0x20 || v8 < 0 )
          goto LABEL_6;
        *(_BYTE *)(v7 + 8) |= 8u;
        v10 = sub_E807D0(*(_QWORD *)(v7 + 24));
        *(_QWORD *)v7 = v10;
        v9 = v10;
        if ( v10 )
          break;
        if ( v6 == ++v4 )
        {
LABEL_14:
          v3 = *(_QWORD *)(a1 + 296);
          goto LABEL_15;
        }
      }
      if ( off_4C5D170 != v9 && (*(_BYTE *)(v7 + 9) & 0x70) != 0x20 && (*(_BYTE *)(v7 + 13) & 2) == 0 )
        break;
LABEL_6:
      if ( v6 == ++v4 )
        goto LABEL_14;
    }
    if ( v68 )
    {
      v1 = v68 - 1;
      v44 = 1;
      v45 = v1 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v46 = (__int64 *)(v66 + 16LL * v45);
      v47 = 0;
      v48 = (void *)*v46;
      if ( (void *)*v46 == v9 )
      {
LABEL_48:
        v49 = v46 + 1;
LABEL_49:
        *v49 = v7;
        goto LABEL_6;
      }
      while ( v48 != (void *)-4096LL )
      {
        if ( !v47 && v48 == (void *)-8192LL )
          v47 = v46;
        v2 = (unsigned int)(v44 + 1);
        v45 = v1 & (v44 + v45);
        v46 = (__int64 *)(v66 + 16LL * v45);
        v48 = (void *)*v46;
        if ( (void *)*v46 == v9 )
          goto LABEL_48;
        ++v44;
      }
      if ( !v47 )
        v47 = v46;
      ++v65;
      v56 = v67 + 1;
      if ( 4 * ((int)v67 + 1) < 3 * v68 )
      {
        if ( v68 - HIDWORD(v67) - v56 <= v68 >> 3 )
        {
          sub_E84990((__int64)&v65, v68);
          if ( !v68 )
          {
LABEL_90:
            LODWORD(v67) = v67 + 1;
            BUG();
          }
          v1 = v66;
          v2 = 0;
          LODWORD(v57) = (v68 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v56 = v67 + 1;
          v58 = 1;
          v47 = (__int64 *)(v66 + 16LL * (unsigned int)v57);
          v59 = (void *)*v47;
          if ( (void *)*v47 != v9 )
          {
            while ( v59 != (void *)-4096LL )
            {
              if ( !v2 && v59 == (void *)-8192LL )
                v2 = (__int64)v47;
              v57 = (v68 - 1) & ((_DWORD)v57 + v58);
              v47 = (__int64 *)(v66 + 16 * v57);
              v59 = (void *)*v47;
              if ( (void *)*v47 == v9 )
                goto LABEL_56;
              ++v58;
            }
            if ( v2 )
              v47 = (__int64 *)v2;
          }
        }
        goto LABEL_56;
      }
    }
    else
    {
      ++v65;
    }
    sub_E84990((__int64)&v65, 2 * v68);
    if ( !v68 )
      goto LABEL_90;
    v2 = v68 - 1;
    LODWORD(v55) = v2 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v56 = v67 + 1;
    v47 = (__int64 *)(v66 + 16LL * (unsigned int)v55);
    v1 = *v47;
    if ( v9 != (void *)*v47 )
    {
      v60 = 1;
      v61 = 0;
      while ( v1 != -4096 )
      {
        if ( v1 == -8192 && !v61 )
          v61 = v47;
        v55 = (unsigned int)v2 & ((_DWORD)v55 + v60);
        v47 = (__int64 *)(v66 + 16 * v55);
        v1 = *v47;
        if ( (void *)*v47 == v9 )
          goto LABEL_56;
        ++v60;
      }
      if ( v61 )
        v47 = v61;
    }
LABEL_56:
    LODWORD(v67) = v56;
    if ( *v47 != -4096 )
      --HIDWORD(v67);
    *v47 = (__int64)v9;
    v49 = v47 + 1;
    *v49 = 0;
    goto LABEL_49;
  }
LABEL_15:
  v11 = *(_QWORD *)(v3 + 40);
  v12 = *(unsigned int *)(v3 + 48);
  v13 = v11 + 8 * v12;
  v63 = (_QWORD *)v13;
  if ( v13 != v11 )
  {
    v14 = *(_QWORD **)(v3 + 40);
    do
    {
      v15 = *v14;
      v16 = 0;
      sub_E96340(*v14);
      v17 = 0;
      v18 = **(_QWORD ***)(v15 + 8);
      if ( v18 )
      {
        while ( 1 )
        {
          if ( v68 )
          {
            v19 = (v68 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
            v20 = (_QWORD *)(v66 + 16LL * v19);
            v21 = (_QWORD *)*v20;
            if ( v18 == (_QWORD *)*v20 )
            {
LABEL_22:
              v22 = v20[1];
              if ( v22 )
                v16 = v22;
            }
            else
            {
              v42 = 1;
              while ( v21 != (_QWORD *)-4096LL )
              {
                v43 = v42 + 1;
                v19 = (v68 - 1) & (v42 + v19);
                v20 = (_QWORD *)(v66 + 16LL * v19);
                v21 = (_QWORD *)*v20;
                if ( v18 == (_QWORD *)*v20 )
                  goto LABEL_22;
                v42 = v43;
              }
            }
          }
          sub_E96410(v15, v17, v16, v13, v66);
          v18 = (_QWORD *)*v18;
          if ( !v18 )
            break;
          ++v17;
        }
      }
      ++v14;
    }
    while ( v63 != v14 );
    v3 = *(_QWORD *)(a1 + 296);
  }
  v23 = *(_QWORD *)(v3 + 24);
  v24 = *(unsigned int *)(v23 + 96);
  if ( (_DWORD)v24 )
  {
    v25 = *(_QWORD **)(v23 + 88);
    v64 = (__int64 *)v3;
    v26 = v3;
    v27 = &v25[3 * v24];
    while ( 1 )
    {
      v28 = *(_QWORD *)(*v25 + 16LL);
      if ( (unsigned __int8)sub_E5CB20(v26, v28, v12, v13, v1, v2) )
        *(_BYTE *)(v28 + 8) |= 0x20u;
      v33 = *(_QWORD *)(v25[1] + 16LL);
      if ( (unsigned __int8)sub_E5CB20(*(_QWORD *)(a1 + 296), v33, v29, v30, v31, v32) )
        *(_BYTE *)(v33 + 8) |= 0x20u;
      v25 += 3;
      if ( v27 == v25 )
        break;
      v26 = *(_QWORD *)(a1 + 296);
    }
    v34 = sub_E6D970(*v64, (__int64)"__LLVM", 6, "__cg_profile", (void *)0xC, 0, 0, 0, 0);
    sub_E844A0(a1, v34, 0);
    v37 = *(_QWORD *)(v34 + 8);
    v38 = *(_QWORD **)v37;
    v39 = 16LL * *(unsigned int *)(v23 + 96);
    v40 = *(_QWORD *)(*(_QWORD *)v37 + 48LL);
    if ( v39 + v40 > *(_QWORD *)(*(_QWORD *)v37 + 56LL) )
    {
      sub_C8D290((__int64)(v38 + 5), v38 + 8, v39 + v40, 1u, v35, v36);
      v40 = v38[6];
    }
    if ( v39 )
    {
      memset((void *)(v38[5] + v40), 0, v39);
      v40 = v38[6];
    }
    v38[6] = v40 + v39;
    v3 = *(_QWORD *)(a1 + 296);
    v23 = *(_QWORD *)(v3 + 24);
  }
  if ( *(_BYTE *)(v23 + 80) )
  {
    v50 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v3 + 168LL) + 680LL);
    sub_E844A0(a1, v50, 0);
    v53 = **(_QWORD ***)(v50 + 8);
    v54 = v53[6];
    if ( (unsigned __int64)(v54 + 8) > v53[7] )
    {
      sub_C8D290((__int64)(v53 + 5), v53 + 8, v54 + 8, 1u, v51, v52);
      v54 = v53[6];
    }
    *(_QWORD *)(v53[5] + v54) = 0;
    v53[6] += 8LL;
  }
  sub_E8AC70(a1);
  return sub_C7D6A0(v66, 16LL * v68, 8);
}
